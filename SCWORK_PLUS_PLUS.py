#!/usr/bin/env python
# coding: utf-8
import argparse
import pandas as pd
from pandas.core.frame import DataFrame
import logging
import os
import sys
import time
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import DaNN.mmd as mmd
import scanpypip.preprocessing as pp
import mytrainers as t
import Myutils as ut
from captum.attr import IntegratedGradients
from models import (AEBase, DaNN, PretrainedPredictor,
                    PretrainedVAEPredictor, VAEBase)
from scipy.spatial import distance_matrix, minkowski_distance, distance
import random
from sklearn.metrics import (average_precision_score, classification_report,
                             mean_squared_error, r2_score, roc_auc_score,
                             confusion_matrix, precision_recall_curve, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



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


def plot_roc_curve(y_true, y_scores, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(output_path)
    plt.show()

def plot_precision_recall_curve(y_true, y_scores, output_path):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, color='green', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(output_path)
    plt.show()

DATA_MAP = {
    "GSE165318": "data/GSE165318_normalized.single.cell.transposed.txt",
}

class TargetModel(nn.Module):
    def __init__(self, source_predcitor, target_encoder):
        super(TargetModel, self).__init__()
        self.source_predcitor = source_predcitor
        self.target_encoder = target_encoder

    def forward(self, X_target, C_target=None):
        if(type(C_target) == type(None)):
            x_tar = self.target_encoder.encode(X_target)
        else:
            x_tar = self.target_encoder.encode(X_target, C_target)
        y_src = self.source_predcitor.predictor(x_tar)
        return y_src

def run_main(args):
    ################################################# START SECTION OF LOADING PARAMETERS #################################################
    t0 = time.time()

    selected_model = None  # Set a default value   

    if(args.checkpoint not in ["False", "True"]):
        selected_model = args.checkpoint
        split_name = selected_model.split("/")[-1].split("_")
        para_names = (split_name[1::2])
        paras = (split_name[0::2])
        args.bulk_h_dims = paras[4]
        args.sc_h_dims = paras[4]
        args.predictor_h_dims = paras[5]
        args.bottleneck = int(paras[3])
        args.drug = paras[2]
        args.dropout = float(paras[7])
        args.dimreduce = paras[6]

        if(paras[0].find("GSE16518") >= 0):
            args.sc_data = "GSE16518"
            args.batch_id = paras[1].split("GSE16518")[1]
        elif(paras[0].find("MIX-Seq") >= 0):
            args.sc_data = "MIX-Seq"
            args.batch_id = paras[1].split("MIX-Seq")[1]
        else:
            args.sc_data = paras[1]

    epochs = args.epochs
    dim_au_out = args.bottleneck
    na = args.missing_value
    if args.sc_data == 'GSE16518':
        data_path = DATA_MAP['GSE16518']
    else:
        data_path = args.sc_data
    test_size = args.test_size
    select_dose = args.dose.upper()
    freeze = args.freeze_pretrain
    valid_size = args.valid_size
    g_disperson = args.var_genes_disp
    min_n_genes = args.min_n_genes
    max_n_genes = args.max_n_genes
    log_path = args.logging_file
    batch_size = args.batch_size
    encoder_hdims = args.bulk_h_dims.split(",")
    encoder_hdims = list(map(int, encoder_hdims))
    data_name = args.sc_data
    label_path = args.label
    reduce_model = args.dimreduce
    predict_hdims = args.predictor_h_dims.split(",")
    predict_hdims = list(map(int, predict_hdims))
    leiden_res = args.cluster_res
    load_model = bool(args.load_sc_model)
    mod = args.mod
    
    para = str(args.bulk) + "_data_" + str(args.sc_data) + "dose" + str(args.dose) + "_bottle_" + str(args.bottleneck) + "_edim_" + str(args.bulk_h_dims) + "_pdim_" + str(args.predictor_h_dims) + "_model_" + reduce_model + "_dropout_" + str(args.dropout) + "_gene_" + str(args.printgene) + "_lr_" + str(args.lr) + "_mod_" + str(args.mod) + "_sam_" + str(args.sampling)
    source_data_path = args.bulk_data

    now = time.strftime("%Y-%m-%d-%H-%M-%S")
    out_path = log_path + now + "transfer.err"
    log_path = log_path + now + "transfer.log"
    out = open(out_path, "w")
    sys.stderr = out

    logging.basicConfig(level=logging.INFO,
                        filename=log_path,
                        filemode='a',
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.info(args)
    logging.info("Start at " + str(t0))

    for path in [args.logging_file, args.bulk_model_path, args.sc_model_path, args.sc_encoder_path, "save/adata/"]:
        if not os.path.exists(path):
            os.makedirs(path)
            print("The new directory is created!")

    if(args.checkpoint not in ["False", "True"]):
        para = os.path.basename(selected_model).split("_DaNN.pkl")[0]
        args.checkpoint = "True"

    sc_encoder_path = args.sc_encoder_path + para
    source_model_path = args.bulk_model_path + para
    print(source_model_path)
    target_model_path = args.sc_model_path + para
    args_df = ut.save_arguments(args, now)


    # Load the metadata file
    metadata_path = "data/GSE165318_metadata_singlecell_modified_with_controls_new.csv"
    metadata = pd.read_csv(metadata_path)
    print(f"Loaded metadata with shape: {metadata.shape}")

    print(metadata['sensitive'].unique())
    
    # Ensure 'CellID' in metadata is of type string
    metadata['CellID'] = metadata['CellID'].astype(str)

    # Convert 'sensitive' to categorical for safe handling
    metadata['sensitive'] = metadata['sensitive'].astype('category')

    ################################################# END SECTION OF LOADING PARAMETERS ##############################################################

    ################################################# START SECTION OF SINGLE CELL DATA REPROCESSING #################################################
    adata = sc.read_text("data/GSE165318_normalized.single.cell.transposed.txt")
    print(f"Initial shape of data loaded from data/GSE165318_normalized.single.cell.transposed.txt: {adata.shape}")


    # Ensure the index of adata.obs is of type string
    adata.obs.index = adata.obs.index.astype(str)

    # Merge the metadata with adata.obs
    adata.obs = adata.obs.merge(metadata, left_index=True, right_on='CellID', how='left')
    print(f"Shape of adata after merging: {adata.shape}")


    # Ensure that the index is set back to 'CellID' and is of string type
    adata.obs.set_index('CellID', inplace=True)
    adata.obs.index = adata.obs.index.astype(str)
    adata.obs['sensitive'] = adata.obs['sensitive'].astype(str)


    # Remove rows where 'sensitive' is 'control'
    adata = adata[adata.obs['sensitive'] != 'control']
    print(f"Shape of adata after removing control samples: {adata.shape}")

    # Convert 'sensitive' column to binary labels
    adata.obs['sensitive_binary'] = adata.obs['sensitive'].map({'sensitive': 1, 'resistant': 0}).astype(int)

    # Now adata.obs['sensitive_binary'] contains the binary labels you need
    print(adata.obs['sensitive_binary'].unique())  # Should show [0, 1]  


    sc.pp.highly_variable_genes(adata, n_top_genes=16383)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    print(f"Shape of data after reducing features: {adata.shape}")

    if adata.shape[1] != 16383:
        logging.warning(f"Feature selection resulted in {adata.shape[1]} features. Adjusting to 16,383.")
        adata = adata[:, np.random.choice(adata.var_names, 16383, replace=False)]
        logging.info(f"Adjusted shape of data: {adata.shape}")

    assert adata.shape[1] == 16383, f"Feature selection failed, expected 16,383 features but got {adata.shape[1]}"

    data = adata.X

    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10)
    sc.tl.leiden(adata, resolution=leiden_res)
    sc.tl.umap(adata)
    adata.obs['leiden_origin'] = adata.obs['leiden']
    adata.obsm['X_umap_origin'] = adata.obsm['X_umap']
    data_c = adata.obs['leiden'].astype("long").tolist()
    ################################################# END SECTION OF SINGLE CELL DATA REPROCESSING ####################################################

    ################################################# START SECTION OF LOADING SC DATA TO THE TENSORS #################################################
    mmscaler = preprocessing.MinMaxScaler()
    data = mmscaler.fit_transform(data)

    adata.obs.index = adata.obs.index.astype(str)

    Xtarget_train, Xtarget_valid, Ctarget_train, Ctarget_valid = train_test_split(data, data_c, test_size=valid_size, random_state=42)

    device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    Xtarget_trainTensor = torch.FloatTensor(Xtarget_train).to(device)
    Xtarget_validTensor = torch.FloatTensor(Xtarget_valid).to(device)
    Ctarget_trainTensor = torch.LongTensor(Ctarget_train).to(device)
    Ctarget_validTensor = torch.LongTensor(Ctarget_valid).to(device)

    X_allTensor = torch.FloatTensor(data).to(device)
    print(f"Data tensor shape: {X_allTensor.shape}")

    C_allTensor = torch.LongTensor(data_c).to(device)
    logging.info(f"Data tensor shape: {X_allTensor.shape}")

    train_dataset = TensorDataset(Xtarget_trainTensor, Ctarget_trainTensor)
    valid_dataset = TensorDataset(Xtarget_validTensor, Ctarget_validTensor)

    Xtarget_trainDataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    Xtarget_validDataLoader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

    dataloaders_pretrain = {'train': Xtarget_trainDataLoader, 'val': Xtarget_validDataLoader}
    ################################################# END SECTION OF LOADING SC DATA TO THE TENSORS #################################################

    ################################################# START SECTION OF LOADING BULK DATA #################################################
    data_r = pd.read_csv(source_data_path, index_col=0)
    label_r = pd.read_csv(label_path, index_col=0)

    print("Shape of data_r:", data_r.shape)
    print("Shape of raw label_r:", label_r.shape)

    label_r = label_r.reindex(data_r.index)
    target_column = label_r.columns[1]
    labels_to_encode = label_r[target_column].fillna('NaN_value').values.reshape(-1, 1)
    print("Shape of labels_to_encode after reshaping:", labels_to_encode.shape)

    le = preprocessing.LabelEncoder()
    encoded_labels = le.fit_transform(labels_to_encode.ravel())
    dim_model_out = len(np.unique(encoded_labels))

    print("Unique values after encoding:", np.unique(encoded_labels))
    print("Encoded labels shape post reshaping and encoding:", encoded_labels.shape)

    assert data_r.shape[0] == encoded_labels.shape[0], "Data and labels must match in samples."

    encoded_labels = torch.LongTensor(encoded_labels).to(device)

    mmscaler = preprocessing.MinMaxScaler()
    source_data = mmscaler.fit_transform(data_r)
    print("Processed source data shape:", source_data.shape)
    print("Encoded labels shape:", encoded_labels.shape)

    Xsource_train_all, Xsource_test, Ysource_train_all, Ysource_test = train_test_split(source_data, encoded_labels, test_size=test_size, random_state=42)
    print(f"Training data shape: {Xsource_train_all.shape}")
    print(f"Test data shape: {Xsource_test.shape}")
    print(f"Training labels shape: {Ysource_train_all.shape}")
    print(f"Test labels shape: {Ysource_test.shape}")

    Xsource_train, Xsource_valid, Ysource_train, Ysource_valid = train_test_split(Xsource_train_all, Ysource_train_all, test_size=valid_size, random_state=42)

    print(f"Ysource_train dtype: {Ysource_train.dtype}")
    print(f"Ysource_train contents: {Ysource_train[:5]}")  # Display the first 5 labels

    Xsource_trainTensor = torch.FloatTensor(Xsource_train).to(device)
    Xsource_validTensor = torch.FloatTensor(Xsource_valid).to(device)
    Ysource_trainTensor = torch.LongTensor(Ysource_train).to(device)
    Ysource_validTensor = torch.LongTensor(Ysource_valid).to(device)

    sourcetrain_dataset = TensorDataset(Xsource_trainTensor, Ysource_trainTensor)
    sourcevalid_dataset = TensorDataset(Xsource_validTensor, Ysource_validTensor)

    Xsource_trainDataLoader = DataLoader(dataset=sourcetrain_dataset, batch_size=batch_size, shuffle=True)
    Xsource_validDataLoader = DataLoader(dataset=sourcevalid_dataset, batch_size=batch_size, shuffle=True)

    dataloaders_source = {'train': Xsource_trainDataLoader, 'val': Xsource_validDataLoader}
    ################################################# END SECTION OF LOADING BULK DATA #################################################

    ################################################# START SECTION OF MODEL CONSTRUCTION #################################################
    if reduce_model == "AE":
        encoder = AEBase(input_dim=adata.shape[1], latent_dim=dim_au_out, h_dims=encoder_hdims, drop_out=args.dropout)
        loss_function_e = nn.MSELoss()
    elif reduce_model == "VAE":
        encoder = VAEBase(input_dim=adata.shape[1], latent_dim=dim_au_out, h_dims=encoder_hdims, drop_out=args.dropout)
    elif reduce_model == "DAE":
        encoder = AEBase(input_dim=adata.shape[1], latent_dim=dim_au_out, h_dims=encoder_hdims, drop_out=args.dropout)
        loss_function_e = nn.MSELoss()

    logging.info(f"Target encoder structure: {encoder}")
    encoder.to(device)
    optimizer_e = optim.Adam(encoder.parameters(), lr=args.lr)
    exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)

    dim_model_out = 2
    if reduce_model in ["AE", "DAE"]:
        source_model = PretrainedPredictor(
            input_dim=dim_au_out,
            latent_dim=dim_au_out,
            h_dims=encoder_hdims,
            hidden_dims_predictor=predict_hdims,
            output_dim=dim_model_out,
            pretrained_weights=None,
            freezed=freeze,
            drop_out=args.dropout,
            drop_out_predictor=args.dropout
        )
    elif reduce_model == "VAE":
        source_model = PretrainedVAEPredictor(
            input_dim=dim_au_out,
            latent_dim=dim_au_out,
            h_dims=encoder_hdims,
            hidden_dims_predictor=predict_hdims,
            output_dim=dim_model_out,
            pretrained_weights=None,
            freezed=freeze,
            z_reparam=bool(args.VAErepram),
            drop_out=args.dropout,
            drop_out_predictor=args.dropout
        )
    source_model.load_state_dict(torch.load(source_model_path, map_location=device, weights_only=True))
    source_encoder = source_model
    source_encoder.to(device)
    ################################################# END SECTION OF MODEL CONSTRUCTION #################################################

    ################################################# START SECTION OF SC MODEL PRETRAINING #################################################
    if(str(args.sc_encoder_path) != 'False'):
        train_flag = True
        sc_encoder_path = str(sc_encoder_path)
        print("Pretrain=="+sc_encoder_path)

        if(args.checkpoint != "False"):
            try:
                encoder.load_state_dict(torch.load(sc_encoder_path, map_location=device, weights_only=True))
                logging.info("Load pretrained target encoder from " + sc_encoder_path)
                train_flag = False
            except:
                logging.info("Loading failed, proceed to re-train model")
                train_flag = True

        if train_flag == True:
            if reduce_model == "AE":
                encoder, loss_report_en = t.train_AE_model(net=encoder, data_loaders=dataloaders_pretrain,
                                                           optimizer=optimizer_e, loss_function=loss_function_e, load=False,
                                                           n_epochs=epochs, scheduler=exp_lr_scheduler_e, save_path=sc_encoder_path)
            if reduce_model == "DAE":
                encoder, loss_report_en = t.train_DAE_model(net=encoder, data_loaders=dataloaders_pretrain,
                                                            optimizer=optimizer_e, loss_function=loss_function_e, load=False,
                                                            n_epochs=epochs, scheduler=exp_lr_scheduler_e, save_path=sc_encoder_path)

            elif reduce_model == "VAE":
                encoder, loss_report_en = t.train_VAE_model(net=encoder, data_loaders=dataloaders_pretrain,
                                                            optimizer=optimizer_e, load=False,
                                                            n_epochs=epochs, scheduler=exp_lr_scheduler_e, save_path=sc_encoder_path)
            logging.info("Pretrained finished")

        embeddings_pretrain = encoder.encode(X_allTensor)
        print(embeddings_pretrain)
        pretrain_prob_prediction = source_model.predict(embeddings_pretrain).detach().cpu().numpy()
        adata.obs["sens_preds_pret"] = pretrain_prob_prediction[:, 1]
        adata.obs["sens_label_pret"] = pretrain_prob_prediction.argmax(axis=1)

        embeddings_pretrain = embeddings_pretrain.detach().cpu().numpy()
        adata.obsm["X_pre"] = embeddings_pretrain
    ################################################# END SECTION OF SC MODEL PRETRAINING #################################################

    ################################################# START SECTION OF TRANSFER LEARNING TRAINING #################################################
    loss_d = nn.CrossEntropyLoss()
    optimizer_d = optim.Adam(encoder.parameters(), lr=1e-2)
    exp_lr_scheduler_d = lr_scheduler.ReduceLROnPlateau(optimizer_d)

    DaNN_model = DaNN(source_model=source_encoder, target_model=encoder, fix_source=bool(args.fix_source))
    DaNN_model.to(device)

    def loss(x, y, GAMMA=args.mmd_GAMMA):
        result = mmd.mmd_loss(x, y, GAMMA)
        return result

    loss_distribution = loss

    logging.info("Training using " + mod + " model")
    target_model = TargetModel(source_model, encoder)
    if selected_model is not None:
        if mod == 'ori':
            if args.checkpoint == 'True':
                DaNN_model, report_ = t.train_DaNN_model(DaNN_model,
                                                         dataloaders_source, dataloaders_pretrain,
                                                         optimizer_d, loss_d,
                                                         epochs, exp_lr_scheduler_d,
                                                         dist_loss=loss_distribution,
                                                         load=target_model_path + "_DaNN.pkl",
                                                         weight=args.mmd_weight,
                                                         save_path=target_model_path + "_DaNN.pkl")
            else:
                DaNN_model, report_ = t.train_DaNN_model(DaNN_model,
                                                         dataloaders_source, dataloaders_pretrain,
                                                         optimizer_d, loss_d,
                                                         epochs, exp_lr_scheduler_d,
                                                         dist_loss=loss_distribution,
                                                         load=False,
                                                         weight=args.mmd_weight,
                                                         save_path=target_model_path + "_DaNN.pkl")
        elif mod == 'new':
            if args.checkpoint == 'True':
                DaNN_model, report_, _, _ = t.train_DaNN_model2(DaNN_model,
                                                                dataloaders_source, dataloaders_pretrain,
                                                                optimizer_d, loss_d,
                                                                epochs, exp_lr_scheduler_d,
                                                                dist_loss=loss_distribution,
                                                                load=selected_model,
                                                                weight=args.mmd_weight,
                                                                save_path=target_model_path + "_DaNN.pkl")
            else:
                DaNN_model, report_, _, _ = t.train_DaNN_model2(DaNN_model,
                                                                dataloaders_source, dataloaders_pretrain,
                                                                optimizer_d, loss_d,
                                                                epochs, exp_lr_scheduler_d,
                                                                dist_loss=loss_distribution,
                                                                load=False,
                                                                weight=args.mmd_weight,
                                                                save_path=target_model_path + "_DaNN.pkl",
                                                                device=device)

    encoder = DaNN_model.target_model
    source_model = DaNN_model.source_model
    logging.info("Transfer DaNN finished")
    ################################################# END SECTION OF TRANSFER LEARNING TRAINING #################################################

    ################################################# START SECTION OF PREPROCESSING FEATURES #################################################
    

    # Encode the data and make predictions
    embedding_tensors = encoder.encode(X_allTensor)
    prediction_tensors = source_model.predictor(embedding_tensors)

    # Apply softmax to get probabilities if it's not already applied in the model definition
    predictions = F.softmax(prediction_tensors, dim=1).detach().cpu().numpy()

    # Debugging: Check the predictions structure
    print("Sample predictions before clipping:")
    print(predictions[:5])
    print("Sum of probabilities per row before normalization:", predictions.sum(axis=1)[:5])

    # Normalize predictions to ensure they sum to 1 row-wise, though softmax should already do this
    epsilon = 1e-7
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    predictions /= predictions.sum(axis=1, keepdims=True)

    print("Sample predictions after clipping and normalization:")
    print(predictions[:5])
    print("Sum of probabilities per row after normalization:", predictions.sum(axis=1)[:5])

    # Set 'sens_pb_results' to the predictions
    sens_pb_results = predictions
    
    # Use predictions in your AnnData object
    adata.obs["sens_preds"] = predictions[:, 1]  # Adjust according to your class index usage
    adata.obs["sens_label"] = predictions.argmax(axis=1)
    adata.obs["sens_label"] = adata.obs["sens_label"].astype('category')
    adata.obs["rest_preds"] = predictions[:, 0]

    ################################################# END SECTION OF ANALYSIS AND POST PROCESSING #################################################

    ################################################# START SECTION OF ANALYSIS FOR scRNA-Seq DATA #################################################
    adata.write("save/adata/" + data_name + para + ".h5ad")
    ################################################# END SECTION OF ANALYSIS FOR scRNA-Seq DATA #################################################

    from sklearn.metrics import (average_precision_score,
                                 classification_report, mean_squared_error, r2_score, roc_auc_score)
    
    report_df = {}
    Y_test = adata.obs['sensitive_binary']
    lb_results = adata.obs['sens_label']

    # Debugging: Ensure that predictions sum to 1
    row_sums = sens_pb_results.sum(axis=1)
    assert np.allclose(row_sums, np.ones_like(row_sums)), "Probabilities do not sum to 1."
    

    # Calculate ROC AUC score for binary classification
    roc_auc = roc_auc_score(Y_test, sens_pb_results[:, 1])
    print(f'ROC AUC: {roc_auc:.4f}')


    
    # Calculate the average precision score for binary classification
    ap_score = average_precision_score(Y_test, sens_pb_results[:, 1])
    print(f'Average Precision Score: {ap_score:.4f}')


    # Generate classification report
    report_dict = classification_report(Y_test, lb_results, output_dict=True)
    f1score = report_dict['weighted avg']['f1-score']

    report_df['f1_score'] = f1score

    file = f'save/bulk_f{data_name}_f1_score_ori.txt'
    with open(file, 'a+') as f:
        f.write(f'{para}\t{f1score}\n')

    print("sc model finished")
    

    cm = confusion_matrix(Y_test, lb_results)

    # Create a figure for the heatmap
    plt.figure(figsize=(10, 7))

    # Create a heatmap with annotations
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])

    # Add titles and labels
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Display the plot
    plt.show()

    # Print the confusion matrix with labels
    print("Confusion Matrix:")
    print("                   Predicted Positive   Predicted Negative")
    print(f"Actual Positive         {cm[1][1]}                  {cm[1][0]}")
    print(f"Actual Negative         {cm[0][1]}                  {cm[0][0]}")

    if (args.printgene == 'T'):
        target_model = TargetModel(source_model, encoder)
        sc_X_allTensor = X_allTensor

        ytarget_allPred = target_model(sc_X_allTensor).detach().cpu().numpy()
        ytarget_allPred = ytarget_allPred.argmax(axis=1)
        ig = IntegratedGradients(target_model)
        scattr, delta = ig.attribute(sc_X_allTensor, target=1, return_convergence_delta=True, internal_batch_size=batch_size)
        scattr = scattr.detach().cpu().numpy()

        igadata = sc.AnnData(scattr)
        igadata.var.index = adata.var.index
        igadata.obs.index = adata.obs.index
        sc_gra = "save/" + data_name + "sc_gradient.txt"
        sc_gen = "save/" + data_name + "sc_gene.csv"
        sc_lab = "save/" + data_name + "sc_label.csv"
        np.savetxt(sc_gra, scattr, delimiter=" ")
        DataFrame(adata.var.index).to_csv(sc_gen)
        DataFrame(adata.obs["sens_label"]).to_csv(sc_lab)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data 
    parser.add_argument('--bulk_data', type=str, default='data/ALL_gene_expression_replicated.csv', help='Path of the bulk RNA-Seq expression profile')
    parser.add_argument('--label', type=str, default='data/reshaped_ALL_label_binary.csv', help='Path of the processed bulk RNA-Seq drug screening annotation')
    parser.add_argument('--sc_data', type=str, default="GSE165318", help='Accession id for testing data, only support pre-built data.')
    parser.add_argument('--dose', type=str, default='ALLDOSES', help='Name of the selected drug, should be a column name in the input file of --label')
    parser.add_argument('--missing_value', type=int, default=1, help='The value filled in the missing entry in the drug screening annotation, default: 1')
    parser.add_argument('--test_size', type=float, default=0.2, help='Size of the test set for the bulk model training, default: 0.2')
    parser.add_argument('--valid_size', type=float, default=0.2, help='Size of the validation set for the bulk model training, default: 0.2')
    parser.add_argument('--var_genes_disp', type=float, default=0, help='Dispersion of highly variable genes selection when pre-processing the data.')
    parser.add_argument('--min_n_genes', type=int, default=0, help="Minimum number of genes for a cell that have UMI counts >1 for filtering purpose, default: 0")
    parser.add_argument('--max_n_genes', type=int, default=20000, help="Maximum number of genes for a cell that have UMI counts >1 for filtering purpose, default: 20000")
    parser.add_argument('--min_g', type=int, default=500, help="Minimum number of genes for a cell >1 for filtering purpose, default: 200")
    parser.add_argument('--min_c', type=int, default=3, help="Minimum number of cells that each gene express for filtering purpose, default: 3")
    parser.add_argument('--percent_mito', type=int, default=100, help="Percentage of expression level of mitochondrial genes of a cell for filtering purpose, default: 100")

    parser.add_argument('--cluster_res', type=float, default=0.2, help="Resolution of Leiden clustering of scRNA-Seq data, default: 0.3")
    parser.add_argument('--mmd_weight', type=float, default=0.25, help="Weight of the MMD loss of the transfer learning, default: 0.25")
    parser.add_argument('--mmd_GAMMA', type=int, default=1000, help="Gamma parameter in the kernel of the MMD loss of the transfer learning, default: 1000")

    # train
    parser.add_argument('--device', type=str, default="cpu", help='Device to train the model. Can be cpu or gpu. Default: cpu')
    parser.add_argument('--bulk_model_path', '-s', type=str, default='save/bulk_pre/', help='Path of the trained predictor in the bulk level')
    parser.add_argument('--sc_model_path', '-p', type=str, default='save/sc_pre/', help='Path (prefix) of the trained predictor in the single cell level')
    parser.add_argument('--sc_encoder_path', type=str, default='save/sc_encoder/', help='Path of the pre-trained encoder in the single-cell level')
    parser.add_argument('--checkpoint', type=str, default='True', help='Load weight from checkpoint files, can be True, False, or a file path.')

    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate of model training. Default: 1e-2')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs for training. Default: 500')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size when training. Default: 200')
    parser.add_argument('--bottleneck', type=int, default=512, help='Size of the bottleneck layer of the model. Default: 32')
    parser.add_argument('--dimreduce', type=str, default="AE", help='Encoder model type. Can be AE or VAE. Default: AE')
    parser.add_argument('--freeze_pretrain', type=int, default=0, help='Fix the parameters in the pretrained model. 0: do not freeze, 1: freeze. Default: 0')
    parser.add_argument('--bulk_h_dims', type=str, default="512,256", help='Shape of the source encoder. Layers are separated by a comma. Default: 512,256')
    parser.add_argument('--sc_h_dims', type=str, default="512,256", help='Shape of the encoder. Layers are separated by a comma. Default: 512,256')
    parser.add_argument('--predictor_h_dims', type=str, default="16,8", help='Shape of the predictor. Layers are separated by a comma. Default: 16,8')
    parser.add_argument('--VAErepram', type=int, default=1)
    parser.add_argument('--batch_id', type=str, default="HN137", help="Batch ID only for testing")
    parser.add_argument('--load_sc_model', type=int, default=0, help='Load a trained model or not. 0: do not load, 1: load. Default: 0')

    parser.add_argument('--mod', type=str, default='new', help='Embed the cell type label to regularize the training: new: add cell type info, ori: do not add cell type info. Default: new')
    parser.add_argument('--printgene', type=str, default='F', help='Print the critical gene list: T: print. Default: F')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout of the neural network. Default: 0.3')

    # Logging and sampling options
    parser.add_argument('--logging_file', '-l', type=str, default='save/log/', help='Path of training log')
    parser.add_argument('--sampling', type=str, default='no', help='Sampling method of training data for the bulk model training. Can be no, upsampling, downsampling, or SMOTE. Default: no')
    parser.add_argument('--fix_source', type=int, default=0, help='Fix the bulk level model. Default: 0')
    parser.add_argument('--bulk', type=str, default='integrate', help='Selection of the bulk database. integrate: both datasets. old: GDSC. new: CCLE. Default: integrate')

    args, unknown = parser.parse_known_args()
    run_main(args)