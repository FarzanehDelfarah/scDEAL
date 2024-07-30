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
import My_utils as ut
import trainers as t
from sklearn.utils.class_weight import compute_class_weight
from Mymodels import AEBase, PretrainedPredictor, PretrainedVAEPredictor, VAEBase
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
    #args.checkpoint = "save/models/data/ALL_gene_expression.csv_data_GSE165318_dose_ALLDOSES_bottle_32_edim_512,256_pdim_16,8_model_AE_dropout_0.3_gene_F_lr_0.01_mod_new_sam_no"
    if(args.checkpoint not in ["False","True"]):
        selected_model = args.checkpoint
        split_name = selected_model.split("/")[-1].split("_")
        para_names = (split_name[1::2])
        paras = (split_name[0::2])
        args.encoder_hdims = paras[4]
        args.predictor_h_dims = paras[5]
        args.bottleneck = int(paras[3])
        args.drug = paras[2]
        args.dropout = float(paras[7])
        args.dimreduce = paras[6]
    # Extract parameters
    epochs = args.epochs
    dim_au_out = args.bottleneck
    select_dose = args.selected_dose  # Use the argument directly without converting to uppercase
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


    # Inspect the unique values in the label_r DataFrame for the select_dose column
    unique_labels = label_r[select_dose].unique()
    print(unique_labels)

    # Check the mapping done by LabelEncoder
    le = LabelEncoder()
    le.fit(unique_labels)
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(label_mapping)



    logging.info(np.std(data))
    logging.info(np.mean(data))

    # Split training, validation, and test sets
    X_train_all, X_test, Y_train_all, Y_test = train_test_split(data, label, test_size=test_size, random_state=42)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_all, Y_train_all, test_size=valid_size, random_state=42)

    # Sampling method
    if sampling == "no":
        X_train, Y_train = sam.nosampling(X_train, Y_train)
    elif sampling == "upsampling":
        X_train, Y_train = sam.upsampling(X_train, Y_train)
    elif sampling == "downsampling":
        X_train, Y_train = sam.downsampling(X_train, Y_train)
    elif sampling == "SMOTE":
        X_train, Y_train = sam.SMOTEsampling(X_train, Y_train)

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
    print("bulk_Y_allTensor", bulk_Y_allTensor.shape)
    


    # Pretrain the encoder if required
    if str(args.pretrain) != "False":
       dataloaders_pretrain = {'train': X_trainDataLoader, 'val': X_validDataLoader}
       if reduce_model == "VAE":
           encoder = VAEBase(input_dim=16382, latent_dim=dim_au_out, h_dims=encoder_hdims, drop_out=args.dropout)
       if reduce_model == 'AE':
            encoder = AEBase(input_dim=16382,latent_dim=dim_au_out,h_dims=encoder_hdims,drop_out=args.dropout)
       if reduce_model in ['DAE']:
           encoder = AEBase(input_dim=16382, latent_dim=dim_au_out, h_dims=encoder_hdims, drop_out=args.dropout)

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

    # Add debugging statement to check the value of reduce_model
    print(f"reduce_model: {reduce_model}")

    # Define the model of predictor
    if reduce_model in ["AE"]:
        model = PretrainedPredictor(
        input_dim=16382,  # Ensure this matches your input data dimensions
        latent_dim=dim_au_out, 
        h_dims=encoder_hdims,
        hidden_dims_predictor=predictor_hdims, 
        output_dim=dim_model_out,
        pretrained_weights=bulk_encoder_path,  # Ensure this is the full path with the filename
        freezed=bool(args.freeze_pretrain), 
        drop_out=args.dropout, 
        drop_out_predictor=args.dropout
    )

    elif reduce_model in ["DAE"]:
        model = PretrainedPredictor(
        input_dim=16382,  # Ensure this matches your input data dimensions
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
        input_dim=16382,  # Ensure this matches your input data dimensions
        latent_dim=dim_au_out, 
        h_dims=encoder_hdims,
        hidden_dims_predictor=predictor_hdims, 
        output_dim=dim_model_out,
        pretrained_weights=bulk_encoder_path,  # Ensure this is the full path with the filename
        freezed=bool(args.freeze_pretrain), 
        drop_out=args.dropout, 
        drop_out_predictor=args.dropout
    )

    else:
        raise ValueError(f"Invalid reduce_model value: {reduce_model}")

    # After the model training and evaluation
    model.eval()

    # Calculate and log performance metrics
    bulk_Y_pre = model.predict(bulk_X_allTensor)
    bulk_Y_pre = bulk_Y_pre.cpu().detach().numpy()
    bulk_Y_pre = np.argmax(bulk_Y_pre, axis=1)
    bulk_Y_allTensor = bulk_Y_allTensor.cpu().detach().numpy()


    print("bulk_Y_allTensor shape:", bulk_Y_allTensor.shape)
    print("bulk_Y_pre shape after argmax:", bulk_Y_pre.shape)

    # Check for dimension mismatch
    if bulk_Y_allTensor.shape != bulk_Y_pre.shape:
        logging.error("Dimension mismatch: bulk_Y_allTensor and bulk_Y_pre have incompatible dimensions.")
        sys.exit(1)

    # Similarly for validation set
    valid_Y_pre = model.predict(X_validTensor)
    valid_Y_pre = valid_Y_pre.cpu().detach().numpy()
    valid_Y_pre = np.argmax(valid_Y_pre, axis=1)
    Y_validTensor = Y_validTensor.cpu().detach().numpy()

    print("Y_validTensor shape:", Y_validTensor.shape)
    print("valid_pre shape after argmax:", valid_Y_pre.shape)

    valid_accuracy = accuracy_score(Y_validTensor, valid_Y_pre)
    valid_precision = precision_score(Y_validTensor, valid_Y_pre, average='weighted')
    valid_recall = recall_score(Y_validTensor, valid_Y_pre, average='weighted')
    valid_f1 = f1_score(Y_validTensor, valid_Y_pre, average='weighted')


    model.to(device)
    bulk_model = "save/models/data/ALL_gene_expression.csv_data_GSE165318_dose_ALLDOSES_bottle_32_edim_512,256_pdim_16,8_model_AE_dropout_0.3_gene_F_lr_0.01_mod_new_sam_no"
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    loss_function = nn.CrossEntropyLoss()
    model, loss_report = t.train_predictor_model(net=model, data_loaders=dataloaders_train,
                                                  optimizer=optimizer, loss_function=loss_function,
                                                  n_epochs=epochs, scheduler=exp_lr_scheduler,
                                                  save_path=bulk_model)

    
    
    
    
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
        bulk_pre = model(bulk_X_allTensor).detach().cpu().numpy()  
        bulk_pre = bulk_pre.argmax(axis=1)
        #print(model)
        #print(bulk_pre.shape)
        # Caculate integrated gradient
        ig = IntegratedGradients(model)
        
        df_results_p = {}
        target=1
        attr, delta =  ig.attribute(bulk_X_allTensor,target=1, return_convergence_delta=True,internal_batch_size=batch_size)
        
        #attr, delta =  ig.attribute(bulk_X_allTensor,target=1, return_convergence_delta=True,internal_batch_size=batch_size)
        attr = attr.detach().cpu().numpy()
        
        np.savetxt("save/"+args.data_name+"bulk_gradient.txt",attr,delimiter = " ")
        from pandas.core.frame import DataFrame
        DataFrame(bulk_pre).to_csv("save/"+args.data_name+"bulk_lab.csv")
    dl_result = model(X_testTensor).detach().cpu().numpy()
    
    
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
    
    

    # Calculate performance metrics
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
    predictions_csv_path = os.path.join(bulk_encoder_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_csv_path, index=False)
    logging.info(f"Predictions saved to {predictions_csv_path}")


    logging.info(f"Validation Accuracy: {valid_accuracy}")
    logging.info(f"Validation Precision: {valid_precision}")
    logging.info(f"Validation Recall: {valid_recall}")
    logging.info(f"Validation F1 Score: {valid_f1}")

    valid_predictions_df = pd.DataFrame({
        'Actual': Y_validTensor,
        'Predicted': valid_Y_pre
    })
    valid_predictions_csv_path = os.path.join(bulk_encoder_dir, 'valid_predictions.csv')
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

    parser.add_argument("--log", type=str, default="save/log/", help="path of log")
    parser.add_argument("--data", type=str, default="data/replicated_ALL_gene_expression.csv", help="path of training data")
    parser.add_argument("--label", type=str, default="data/reshaped_ALL_label_binary.csv", help="path of training label")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=200, help="batch size")
    parser.add_argument("--selected_dose", type=str, default="ALLDOSES", help="Selected dose for training")
    parser.add_argument("--missing_value", type=str, default=0, help="the value to represent missing")
    parser.add_argument("--bottleneck", type=int, default=32, help="the dimension of output of AE or VAE")
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
    parser.add_argument('--printgene', type=str, default="F",help='Print the cirtical gene list: T: print. Default: T')
    parser.add_argument('--data_name', type=str, default="GSE165318",help='Accession id for testing data, only support pre-built data.')
    parser.add_argument("--checkpoint", type=str, default="True", help="whether to use checkpoint")
    parser.add_argument("--freeze_pretrain", type=int, default=1, help="whether to freeze pretrained weights")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
    parser.add_argument("--bulk_model", type=str, default="save/models/data", help="path to save bulk model")
    parser.add_argument("--bulk_encoder", type=str, default="save/encoders/", help="path to save bulk encoder")

    args = parser.parse_args()
    run_main(args)