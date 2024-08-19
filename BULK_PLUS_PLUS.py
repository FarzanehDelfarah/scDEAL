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
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (average_precision_score,
                             classification_report, mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

import sampling as sam
import Myutils as ut
import mytrainers as t
from sklearn.utils.class_weight import compute_class_weight
from models import AEBase, PretrainedPredictor, PretrainedVAEPredictor, VAEBase
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    select_dose = args.selected_dose
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

    # Read data and labels
    data_r = pd.read_csv(data_path, index_col=0, engine='python')
    label_r = pd.read_csv(label_path, index_col=0, engine='python')
    label_r = label_r.fillna(na)

    print(f"Initial shape of data: {data_r.shape}")
    print(f"Initial shape of labels: {label_r.shape}")

    # Initialize logging and std out
    out_path = log_path + now + "bulk.err"
    log_path = log_path + now + "bulk.log"

    # Ensure the number of samples in data and labels match
    assert data_r.shape[0] == label_r.shape[0], "Mismatch in the number of samples between data and labels!"

    out = open(out_path, "w")
    sys.stderr = out

    logging.basicConfig(level=logging.INFO,
                        filename=log_path,
                        filemode='a',
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.info(args)

    # Filter out na values
    selected_idx = label_r.loc[:,select_dose]!=na

    # Apply HVG filtering for bulk data
    if g_disperson is not None:
        print(f"Applying HVG filtering with min_disp = {g_disperson}")
        hvg, bulk_adata = ut.highly_variable_genes(data_r, min_disp=g_disperson, for_bulk=True)
        data_r = data_r.loc[:, hvg]  # Subset the data based on highly variable genes
    else:
        data_r = data_r.loc[selected_idx, :]

    print(f"Shape after HVG filtering: {data_r.shape}")

    # Ensure number of samples still matches after filtering
    assert data_r.shape[0] == label_r.shape[0], "Mismatch after HVG filtering!"

    # Do PCA if required
    if PCA_dim != 0:
       data_r = PCA(n_components=PCA_dim).fit_transform(data_r)
    else:
       data_r = data_r.values
    print(f"Shape after PCA transformation: {data_r.shape}")

    # Handle labels correctly
    label = label_r.loc[selected_idx, select_dose].values.reshape(-1, 1)
    le = LabelEncoder()
    label = le.fit_transform(label)
    
    # Scale data
    mmscaler = preprocessing.MinMaxScaler()
    data_r = mmscaler.fit_transform(data_r)
    print(f"Shape of data after scaling: {data_r.shape}")

    # Recheck that data and label shapes are consistent
    assert data_r.shape[0] == len(label), "Data and labels must have the same number of samples after preprocessing!"

    dim_model_out = 2

    # Unique labels and label encoding check
    unique_labels = label_r[select_dose].unique()
    print("Unique labels:", unique_labels)

    le = LabelEncoder()
    le.fit(unique_labels)
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print("Label mapping:", label_mapping)

    logging.info(np.std(data_r))
    logging.info(np.mean(data_r))

    # Split the data
    X_train_all, X_test, Y_train_all, Y_test = train_test_split(data_r, label, test_size=test_size, random_state=42)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_all, Y_train_all, test_size=valid_size, random_state=42)

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
        logging.info("Not a legal sampling method")

    # Select the training device
    device = torch.device("cuda:0" if args.device == "gpu" and torch.cuda.is_available() else "cpu")
    print(device)

    # Define the encoder model
    if reduce_model == "VAE":
        encoder = VAEBase(input_dim=16383, latent_dim=dim_au_out, h_dims=encoder_hdims, drop_out=args.dropout)
    elif reduce_model == 'AE':
        encoder = AEBase(input_dim=16383, latent_dim=dim_au_out, h_dims=encoder_hdims, drop_out=args.dropout)
    elif reduce_model == "DAE":
        encoder = AEBase(input_dim=16383, latent_dim=dim_au_out, h_dims=encoder_hdims, drop_out=args.dropout)
    else:
        raise ValueError(f"Invalid reduce_model value: {reduce_model}")

    encoder.to(device)  # Move the encoder to the appropriate device (GPU/CPU)

    # Pretrain the encoder if required
    if str(args.pretrain) != "False":
        dataloaders_pretrain = {'train': DataLoader(TensorDataset(torch.FloatTensor(X_train).to(device), torch.LongTensor(Y_train).to(device)), batch_size=batch_size, shuffle=True),
                                'val': DataLoader(TensorDataset(torch.FloatTensor(X_valid).to(device), torch.LongTensor(Y_valid).to(device)), batch_size=batch_size, shuffle=True)}
        optimizer_e = optim.Adam(encoder.parameters(), lr=1e-2)
        loss_function_e = nn.MSELoss()
        exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)
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
                load=args.checkpoint != "False",
                n_epochs=epochs, 
                scheduler=exp_lr_scheduler_e, 
                save_path=bulk_encoder_path
            )
        elif reduce_model == "VAE":
            encoder, loss_report_en = t.train_VAE_model(
                net=encoder, 
                data_loaders=dataloaders_pretrain,
                optimizer=optimizer_e, 
                load=args.checkpoint != "False",
                n_epochs=epochs, 
                scheduler=exp_lr_scheduler_e, 
                save_path=bulk_encoder_path
            )
        elif reduce_model == "DAE":
            encoder, loss_report_en = t.train_DAE_model(
                net=encoder, 
                data_loaders=dataloaders_pretrain,
                optimizer=optimizer_e, 
                loss_function=loss_function_e, 
                load=args.checkpoint != "False",
                n_epochs=epochs, 
                scheduler=exp_lr_scheduler_e, 
                save_path=bulk_encoder_path
            )
        logging.info("Pretraining of encoder finished")

    # Convert the training, validation, and test datasets to tensors
    X_trainTensor = torch.FloatTensor(X_train).to(device)
    X_validTensor = torch.FloatTensor(X_valid).to(device)  # Corrected from FloatFloatTensor to FloatTensor
    X_testTensor = torch.FloatTensor(X_test).to(device)    # Corrected from FloatFloatTensor to FloatTensor
    Y_trainTensor = torch.LongTensor(Y_train).to(device)
    Y_validTensor = torch.LongTensor(Y_valid).to(device)
    Y_testTensor = torch.LongTensor(Y_test).to(device)

    # Encode the training and validation data before training
    if reduce_model in ["AE", "DAE"]:
        X_trainTensor_encoded = encoder.encode(X_trainTensor)
        X_validTensor_encoded = encoder.encode(X_validTensor)
        X_testTensor_encoded = encoder.encode(X_testTensor)
    elif reduce_model == "VAE":
        X_trainTensor_encoded = encoder.encode(X_trainTensor, repram=True)
        X_validTensor_encoded = encoder.encode(X_validTensor, repram=True)
        X_testTensor_encoded = encoder.encode(X_testTensor, repram=True)
    else:
        raise ValueError(f"Invalid reduce_model value: {reduce_model}")

    # Log tensor shapes for debugging
    logging.info(f"X_trainTensor_encoded shape: {X_trainTensor_encoded.shape}")
    logging.info(f"X_validTensor_encoded shape: {X_validTensor_encoded.shape}")
    logging.info(f"X_testTensor_encoded shape: {X_testTensor_encoded.shape}")

    # Create DataLoaders for the encoded data
    train_dataset_encoded = TensorDataset(X_trainTensor_encoded, Y_trainTensor)
    valid_dataset_encoded = TensorDataset(X_validTensor_encoded, Y_validTensor)
    test_dataset_encoded = TensorDataset(X_testTensor_encoded, Y_testTensor)

    X_trainDataLoader_encoded = DataLoader(dataset=train_dataset_encoded, batch_size=batch_size, shuffle=True)
    X_validDataLoader_encoded = DataLoader(dataset=valid_dataset_encoded, batch_size=batch_size, shuffle=True)
    X_testDataLoader_encoded = DataLoader(dataset=test_dataset_encoded, batch_size=batch_size, shuffle=False)

    # Define the predictor model
    if reduce_model in ["AE", "DAE"]:
        model = PretrainedPredictor(
            input_dim=dim_au_out,
            latent_dim=dim_au_out,
            h_dims=encoder_hdims,
            hidden_dims_predictor=predictor_hdims,
            output_dim=dim_model_out,
            pretrained_weights=None,  # No pre-trained weights
            freezed=False,  # Don't freeze any layers
            drop_out=args.dropout,
            drop_out_predictor=args.dropout
        )
    elif reduce_model == "VAE":
        model = PretrainedVAEPredictor(
            input_dim=dim_au_out,
            latent_dim=dim_au_out,
            h_dims=encoder_hdims,
            hidden_dims_predictor=predictor_hdims,
            output_dim=dim_model_out,
            pretrained_weights=None,  # No pre-trained weights
            freezed=False,  # Don't freeze any layers
            drop_out=args.dropout,
            drop_out_predictor=args.dropout
        )
    else:
        raise ValueError(f"Invalid reduce_model value: {reduce_model}")



    # Evaluate the model on the bulk data
    bulk_X_allTensor = torch.FloatTensor(data_r).to(device)
    bulk_Y_allTensor = torch.LongTensor(label).to(device)
    if reduce_model in ["AE", "DAE"]:
        bulk_X_allTensor_encoded = encoder.encode(bulk_X_allTensor)
    elif reduce_model == "VAE":
        bulk_X_allTensor_encoded = encoder.encode(bulk_X_allTensor, repram=True)
    else:
        raise ValueError(f"Invalid reduce_model value: {reduce_model}")



    model.eval()


    bulk_Y_pre = model.predict(bulk_X_allTensor_encoded)
    bulk_Y_pre = bulk_Y_pre.cpu().detach().numpy()
    bulk_Y_pre = np.argmax(bulk_Y_pre, axis=1)
    bulk_Y_allTensor = bulk_Y_allTensor.cpu().detach().numpy()

    print("bulk_Y_allTensor shape:", bulk_Y_allTensor.shape)
    print("bulk_Y_pre shape after argmax:", bulk_Y_pre.shape)

    # Check for dimension mismatch
    if bulk_Y_allTensor.shape != bulk_Y_pre.shape:
        logging.error("Dimension mismatch: bulk_Y_allTensor and bulk_Y_pre have incompatible dimensions.")
        sys.exit(1)



    # Evaluate on validation set
    valid_Y_pre = model.predict(X_validTensor_encoded)
    valid_Y_pre = valid_Y_pre.cpu().detach().numpy()
    valid_Y_pre = np.argmax(valid_Y_pre, axis=1)
    Y_validTensor = Y_validTensor.cpu().detach().numpy()





    model.to(device)
    # Train the predictor model
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    bulk_model = "save/bulk_pre/model_checkpoint.pth"
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    loss_function = nn.CrossEntropyLoss()

    # Train the predictor model
    model, loss_report = t.train_predictor_model(
        net=model, 
        data_loaders={'train': X_trainDataLoader_encoded, 'val': X_validDataLoader_encoded},
        optimizer=optimizer, 
        loss_function=loss_function,
        n_epochs=epochs, 
        scheduler=exp_lr_scheduler,
        save_path=bulk_model  # Use the corrected save path with a filename
    )



    if (args.printgene=='T'):
        import scanpypip.preprocessing as pp
        bulk_adata = pp.read_sc_file(data_path)
        #print('pp')
        ## bulk test predict critical gene
        import scanpy as sc
        #import scanpypip.utils as uti
        from captum.attr import IntegratedGradients
        #bulk_adata = bulk_adata
        #print(bulk_adata) 
        bulk_pre = model(bulk_X_allTensor_encoded).detach().cpu().numpy()  
        bulk_pre = bulk_pre.argmax(axis=1)
        #print(model)
        #print(bulk_pre.shape)
        # Caculate integrated gradient
        ig = IntegratedGradients(model)
        
        df_results_p = {}
        target=1
        attr, delta =  ig.attribute(bulk_X_allTensor_encoded,target=1, return_convergence_delta=True,internal_batch_size=batch_size)
        
        #attr, delta =  ig.attribute(bulk_X_allTensor,target=1, return_convergence_delta=True,internal_batch_size=batch_size)
        attr = attr.detach().cpu().numpy()
        
        np.savetxt("save/"+args.data_name+"bulk_gradient.txt",attr,delimiter = " ")
        from pandas.core.frame import DataFrame
        DataFrame(bulk_pre).to_csv("save/"+args.data_name+"bulk_lab.csv")
    dl_result = model(X_testTensor_encoded).detach().cpu().numpy()
    
    
    lb_results = np.argmax(dl_result,axis=1)
    #pb_results = np.max(dl_result,axis=1)
    pb_results = dl_result[:,1]

    report_dict = classification_report(Y_test, lb_results, output_dict=True)
    report_df = pd.DataFrame(report_dict).T
    ap_score = average_precision_score(Y_test, pb_results)
    auroc_score = roc_auc_score(Y_test, pb_results)

    report_df['auroc_score'] = auroc_score
    report_df['ap_score'] = ap_score

    report_df.to_csv("save/log/" + reduce_model + select_dose+now + '_report.csv')

    #logging.info(classification_report(Y_test, lb_results))
    #logging.info(average_precision_score(Y_test, pb_results))
    #logging.info(roc_auc_score(Y_test, pb_results))

    model = DummyClassifier(strategy='stratified')
    model.fit(X_train, Y_train)
    yhat = model.predict_proba(X_test)
    naive_probs = yhat[:, 1]

    # ut.plot_roc_curve(Y_test, naive_probs, pb_results, title=str(roc_auc_score(Y_test, pb_results)),
    #                     path="save/figures/" + reduce_model + select_drug+now + '_roc.pdf')
    # ut.plot_pr_curve(Y_test,pb_results,  title=average_precision_score(Y_test, pb_results),
    #                 path="save/figures/" + reduce_model + select_drug+now + '_prc.pdf')
    print("bulk_model finished")


    # Calculate and log performance metrics
    accuracy = accuracy_score(bulk_Y_allTensor, bulk_Y_pre)
    precision = precision_score(bulk_Y_allTensor, bulk_Y_pre, average='weighted')
    recall = recall_score(bulk_Y_allTensor, bulk_Y_pre, average='weighted')
    f1 = f1_score(bulk_Y_allTensor, bulk_Y_pre, average='weighted')

    # Save results
    np.save(f'save/ori_result/{select_dose}_bulk_Y_pre.npy', bulk_Y_pre)
    np.save(f'save/ori_result/{select_dose}_bulk_Y.npy', bulk_Y_allTensor)

    # Log the performance metrics
    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")
    logging.info(f"F1 Score: {f1}")

    # Save predictions and actual labels to CSV
    predictions_df = pd.DataFrame({
        'Actual': bulk_Y_allTensor,
        'Predicted': bulk_Y_pre
    })
    predictions_csv_path = os.path.join(args.bulk_encoder, 'predictions.csv')
    predictions_df.to_csv(predictions_csv_path, index=False)
    logging.info(f"Predictions saved to {predictions_csv_path}")    


    valid_accuracy = accuracy_score(Y_validTensor, valid_Y_pre)
    valid_precision = precision_score(Y_validTensor, valid_Y_pre, average='weighted')
    valid_recall = recall_score(Y_validTensor, valid_Y_pre, average='weighted')
    valid_f1 = f1_score(Y_validTensor, valid_Y_pre, average='weighted')

    # Log validation metrics
    logging.info(f"Validation Accuracy: {valid_accuracy}")
    logging.info(f"Validation Precision: {valid_precision}")
    logging.info(f"Validation Recall: {valid_recall}")
    logging.info(f"Validation F1 Score: {valid_f1}")



    # Save validation predictions
    valid_predictions_df = pd.DataFrame({
        'Actual': Y_validTensor,
        'Predicted': valid_Y_pre
    })
    valid_predictions_csv_path = os.path.join(args.bulk_encoder, 'valid_predictions.csv')
    valid_predictions_df.to_csv(valid_predictions_csv_path, index=False)
    logging.info(f"Validation predictions saved to {valid_predictions_csv_path}")

    print("Training and evaluation complete.")




    accuracy = np.sum(valid_Y_pre == Y_validTensor) / len(Y_validTensor)
    precision = average_precision_score(Y_validTensor, valid_Y_pre)
    logging.info(f"accuracy: {accuracy}")
    logging.info(f"precision: {precision}")
    logging.info(f"\n {classification_report(Y_validTensor, valid_Y_pre)}")


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()

    # Add argument definitions...
    parser.add_argument("--log", type=str, default="save/log/", help="path of log")
    parser.add_argument("--data", type=str, default="data/ALL_gene_expression_replicated.csv", help="path of training data")
    parser.add_argument("--label", type=str, default="data/reshaped_ALL_label_binary.csv", help="path of training label")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=200, help="batch size")
    parser.add_argument("--selected_dose", type=str, default="ALLDOSES", help="Selected dose for training")
    parser.add_argument("--missing_value", type=str, default=0, help="the value to represent missing")
    parser.add_argument("--bottleneck", type=int, default=512, help="the dimension of output of AE or VAE")
    parser.add_argument("--var_genes_disp", type=float, default=None, help="minimum value of dispersion to choose HVGs")
    parser.add_argument("--valid_size", type=float, default=0.2, help="validation size")
    parser.add_argument("--test_size", type=float, default=0.2, help="test size")
    parser.add_argument("--encoder_h_dims", type=str, default="512,256", help="hidden dimension of encoder")
    parser.add_argument("--predictor_h_dims", type=str, default="16,8", help="hidden dimension of predictor")
    parser.add_argument("--dimreduce", type=str, default="AE", help="model to reduce dimension, VAE, AE or DAE")
    parser.add_argument("--sampling", type=str, default="no", help="sampling method")
    parser.add_argument("--PCA_dim", type=int, default=0, help="number of PCA dimensions")
    parser.add_argument("--device", type=str, default="cpu", help="use gpu or cpu")
    parser.add_argument("--pretrain", type=str, default="True", help="whether to pretrain encoder")
    parser.add_argument('--VAErepram', type=int, default=1)
    parser.add_argument('--printgene', type=str, default="F", help='Print the critical gene list: T: print. Default: F')
    parser.add_argument('--data_name', type=str, default="GSE165318", help='Accession id for testing data, only support pre-built data.')
    parser.add_argument("--checkpoint", type=str, default="False", help="whether to use checkpoint")
    parser.add_argument("--freeze_pretrain", type=int, default=1, help="whether to freeze pretrained weights")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
    parser.add_argument("--bulk_model", type=str, default="save/bulk_pre/", help="path to save bulk model")
    parser.add_argument("--bulk_encoder", type=str, default="save/encoders/", help="path to save bulk encoder")

    args = parser.parse_args()
    run_main(args)