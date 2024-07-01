import argparse
import logging
import sys
import time
import warnings
import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.metrics import (average_precision_score, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

import sampling as sam
import utils as ut
import trainers as t
from modified_model import AEBase, PretrainedPredictor, PretrainedVAEPredictor, VAEBase
import matplotlib
import random

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def run_main(args):
    # Extract parameters
    epochs = args.epochs
    dim_au_out = args.bottleneck
    select_dose = args.dose  # Use the argument directly without converting to uppercase
    na = args.missing_value
    data_path = args.data
    label_path = args.label
    test_size = args.test_size
    valid_size = args.valid_size
    g_disperson = args.var_genes_disp
    log_path = args.log
    batch_size = args.batch_size
    encoder_hdims = list(map(int, args.encoder_h_dims.split(",")))
    predictor_hdims = list(map(int, args.predictor_h_dims.split(",")))
    reduce_model = args.dimreduce
    sampling = args.sampling
    PCA_dim = args.PCA_dim

    now = time.strftime("%Y-%m-%d-%H-%M-%S")
    for path in [args.log, args.bulk_model, args.bulk_encoder, 'save/ori_result', 'save/figures']:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"The new directory {path} is created!")

    # Read data
    data_r = pd.read_csv(data_path, index_col=0, engine='python')
    label_r = pd.read_csv(label_path, index_col=0, engine='python')
    label_r = label_r.fillna(na)

    # Initialize logging and std out
    out_path = log_path + now + "bulk.err"
    log_path = log_path + now + "bulk.log"

    out = open(out_path, "w")
    sys.stderr = out

    logging.basicConfig(level=logging.INFO,
                        filename=log_path,
                        filemode='a',
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.info(args)

    # Filter out na values
    selected_idx = data_r.index  # Include all rows

    if g_disperson is not None:
        hvg, adata = ut.highly_variable_genes(data_r, min_disp=g_disperson)
        data_r.columns = adata.var_names
        data = data_r.loc[selected_idx, hvg]
    else:
        data = data_r.loc[selected_idx, :]

    # Do PCA if PCA_dim != 0
    if PCA_dim != 0:
        data = PCA(n_components=PCA_dim).fit_transform(data)
    else:
        data = data.values

    # Extract labels
    label = label_r.loc[selected_idx, select_dose]
    data_r = data_r.loc[selected_idx, :]

    # Scaling data
    mmscaler = preprocessing.MinMaxScaler()
    data = mmscaler.fit_transform(data)
    label = label.values.reshape(-1, 1)

    le = LabelEncoder()
    label = le.fit_transform(label)
    dim_model_out = 2

    logging.info(np.std(data))
    logging.info(np.mean(data))

    # Split training, validation, and test sets
    X_train_all, X_test, Y_train_all, Y_test = train_test_split(data, label, test_size=test_size, random_state=42)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_all, Y_train_all, test_size=valid_size, random_state=42)

    # Sampling method
    if sampling == "no":
        X_train, Y_train = sam.nosampling(X_train, Y_train)
        logging.info("nosampling")
    elif sampling == "upsampling":
        X_train, Y_train = sam.upsampling(X_train, Y_train)
        logging.info("upsampling")
    elif sampling == "downsampling":
        X_train, Y_train = sam.downsampling(X_train, Y_train)
        logging.info("downsampling")
    elif sampling == "SMOTE":
        X_train, Y_train = sam.SMOTEsampling(X_train, Y_train)
        logging.info("SMOTE")
    else:
        logging.info("not a legal sampling method")

    # Select the training device
    device = torch.device("cuda:0" if args.device == "gpu" and torch.cuda.is_available() else "cpu")
    print(device)

    # Construct datasets and data loaders
    X_trainTensor = torch.FloatTensor(X_train).to(device)
    X_validTensor = torch.FloatTensor(X_valid).to(device)
    X_testTensor = torch.FloatTensor(X_test).to(device)

    Y_trainTensor = torch.LongTensor(Y_train).to(device)
    Y_validTensor = torch.LongTensor(Y_valid).to(device)

    train_dataset = TensorDataset(X_trainTensor, Y_trainTensor)
    valid_dataset = TensorDataset(X_validTensor, Y_validTensor)

    X_trainDataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    X_validDataLoader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

    trainreducedDataset = TensorDataset(X_trainTensor, Y_trainTensor)
    validreducedDataset = TensorDataset(X_validTensor, Y_validTensor)

    trainDataLoader_p = DataLoader(dataset=trainreducedDataset, batch_size=batch_size, shuffle=True)
    validDataLoader_p = DataLoader(dataset=validreducedDataset, batch_size=batch_size, shuffle=True)
    bulk_X_allTensor = torch.FloatTensor(data).to(device)
    bulk_Y_allTensor = torch.LongTensor(label).to(device)
    dataloaders_train = {'train': trainDataLoader_p, 'val': validDataLoader_p}
    print("bulk_X_allTensor", bulk_X_allTensor.shape)

    # Pretrain the encoder if required
    if str(args.pretrain) != "False":
        dataloaders_pretrain = {'train': X_trainDataLoader, 'val': X_validDataLoader}
        if reduce_model == "VAE":
            encoder = VAEBase(input_dim=data.shape[1], latent_dim=dim_au_out, h_dims=encoder_hdims, drop_out=args.dropout)
        elif reduce_model in ['AE', 'DAE']:
            encoder = AEBase(input_dim=data.shape[1], latent_dim=dim_au_out, h_dims=encoder_hdims, drop_out=args.dropout)

        encoder.to(device)
        optimizer_e = optim.Adam(encoder.parameters(), lr=1e-2)
        loss_function_e = nn.MSELoss()
        exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)
        bulk_encoder = "save/encoders/"
        load = bulk_encoder if args.checkpoint != "False" else False

        bulk_encoder_dir = "save/encoders/"
        if not os.path.exists(bulk_encoder_dir):
            os.makedirs(bulk_encoder_dir)

        bulk_encoder_path = os.path.join(bulk_encoder_dir, "encoder_model.pth")

        load = bulk_encoder_path if args.checkpoint != "False" else False

        if reduce_model == "AE":
           encoder, loss_report_en = t.train_AE_model(
               net=encoder, 
               data_loaders=dataloaders_pretrain,
               optimizer=optimizer_e, 
               loss_function=loss_function_e, 
               load=load,
               n_epochs=epochs, 
               scheduler=exp_lr_scheduler_e, 
               save_path=bulk_encoder_path  # Use the full path with a filename
           )


        elif reduce_model == "VAE":
            encoder, loss_report_en = t.train_VAE_model(
                net=encoder, 
                data_loaders=dataloaders_pretrain,
                optimizer=optimizer_e, 
                load=False,
                n_epochs=epochs, 
                scheduler=exp_lr_scheduler_e, 
                save_path=bulk_encoder_path  # Use the full path with a filename
            )

        elif reduce_model == "DAE":
            encoder, loss_report_en = t.train_DAE_model(
                net=encoder, 
                data_loaders=dataloaders_pretrain,
                optimizer=optimizer_e, 
                loss_function=loss_function_e, 
                load=load,
                n_epochs=epochs, 
                scheduler=exp_lr_scheduler_e, 
                save_path=bulk_encoder_path  # Use the full path with a filename
            )
        logging.info("Pretrained finished")

    # Define the model of predictor
        if reduce_model in ["AE", "DAE"]:
           model = PretrainedPredictor(
               input_dim=X_train.shape[1], 
               latent_dim=dim_au_out, 
               h_dims=encoder_hdims,
               hidden_dims_predictor=predictor_hdims, 
               output_dim=dim_model_out,
               pretrained_weights=bulk_encoder_path,  # Ensure this is the full path with the filename
               freezed=bool(args.freeze_pretrain), 
               drop_out=args.dropout, 
               drop_out_predictor=args.dropout
            )
        elif reduce_model == "VAE":
            model = PretrainedVAEPredictor(
                input_dim=X_train.shape[1], 
                latent_dim=dim_au_out, 
                h_dims=encoder_hdims,
                hidden_dims_predictor=predictor_hdims, 
                output_dim=dim_model_out,
                pretrained_weights=bulk_encoder_path,  # Ensure this is the full path with the filename
                freezed=bool(args.freeze_pretrain), 
                drop_out=args.dropout, 
                drop_out_predictor=args.dropout
            )
        elif reduce_model == "no":
            model = ut.construct_predictor(
                input_dim=X_train.shape[1], 
                hidden_dims_predictor=predictor_hdims,
                output_dim=dim_model_out, 
                drop_out=args.dropout
            )

    model.to(device)
    bulk_model = "save/models/bulk_model.pth"
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    loss_function = nn.CrossEntropyLoss()
    model, loss_report = t.train_predictor_model(net=model, data_loaders=dataloaders_train,
                                                  optimizer=optimizer, loss_function=loss_function,
                                                  n_epochs=epochs, scheduler=exp_lr_scheduler,
                                                  save_path=bulk_model)

    model.eval()
    bulk_Y_pre = model.predict(bulk_X_allTensor)
    bulk_Y_pre = bulk_Y_pre.cpu().detach().numpy()
    bulk_Y_allTensor = bulk_Y_allTensor.cpu().detach().numpy()

    logging.info(np.unique(bulk_Y_pre))
    logging.info(np.unique(bulk_Y_allTensor))


    valid_pre = model.predict(X_validTensor)
    valid_pre = valid_pre.cpu().detach().numpy()
    Y_validTensor = Y_validTensor.cpu().detach().numpy()


    if bulk_X_allTensor.shape[1] != X_train.shape[1]:
        logging.error("Dimension mismatch: bulk_X_allTensor and X_train have incompatible feature dimensions.")
        sys.exit(1)

    if X_validTensor.shape[1] != X_train.shape[1]:
        logging.error("Dimension mismatch: X_validTensor and X_train have incompatible feature dimensions.")
        sys.exit(1)

    if bulk_Y_allTensor.shape != bulk_Y_pre.shape:
        logging.error("Dimension mismatch: bulk_Y_allTensor and bulk_Y_pre have incompatible dimensions.")
        sys.exit(1)

    if Y_validTensor.shape != valid_pre.shape:
        logging.error("Dimension mismatch: Y_validTensor and valid_pre have incompatible dimensions.")
        sys.exit(1)

    np.save(f'save/ori_result/{select_dose}_bulk_Y_pre.npy', bulk_Y_pre)
    np.save(f'save/ori_result/{select_dose}_bulk_Y.npy', bulk_Y_allTensor)
    
    accuracy = np.sum(valid_pre == Y_validTensor) / len(Y_validTensor)
    precision = average_precision_score(Y_validTensor, valid_pre)
    logging.info(f"accuracy: {accuracy}")
    logging.info(f"precision: {precision}")
    logging.info(f"\n {classification_report(Y_validTensor, valid_pre)}")

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()

    parser.add_argument("--log", type=str, default="save/log/", help="path of log")
    parser.add_argument("--data", type=str, default="data/ALL_gene_expression.csv", help="path of training data")
    parser.add_argument("--label", type=str, default="data/ALL_label_binary.csv", help="path of training label")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--dose", type=str, default="DOSE2", help="Selected dose for training")
    parser.add_argument("--missing_value", type=str, default=0, help="the value to represent missing")
    parser.add_argument("--bottleneck", type=int, default=128, help="the dimension of output of AE or VAE")
    parser.add_argument("--var_genes_disp", type=float, default=None, help="minimum value of dispersion to choose HVGs")
    parser.add_argument("--valid_size", type=float, default=0.2, help="validation size")
    parser.add_argument("--test_size", type=float, default=0.1, help="test size")
    parser.add_argument("--encoder_h_dims", type=str, default="256,128", help="hidden dimension of encoder")
    parser.add_argument("--predictor_h_dims", type=str, default="64,16", help="hidden dimension of predictor")
    parser.add_argument("--dimreduce", type=str, default="AE", help="model to reduce dimension, VAE, AE or DAE")
    parser.add_argument("--sampling", type=str, default="no", help="sampling method")
    parser.add_argument("--PCA_dim", type=int, default=0, help="number of PCA dimensions")
    parser.add_argument("--device", type=str, default="cpu", help="use gpu or cpu")
    parser.add_argument("--pretrain", type=str, default="True", help="whether to pretrain encoder")
    parser.add_argument("--checkpoint", type=str, default="False", help="whether to use checkpoint")
    parser.add_argument("--freeze_pretrain", type=int, default=1, help="whether to freeze pretrained weights")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
    parser.add_argument("--bulk_model", type=str, default="save/models/", help="path to save bulk model")
    parser.add_argument("--bulk_encoder", type=str, default="save/encoders/", help="path to save bulk encoder")

    args = parser.parse_args()
    run_main(args)
