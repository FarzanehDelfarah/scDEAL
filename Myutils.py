import logging
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from captum.attr import IntegratedGradients
from scipy.stats import mannwhitneyu
from sklearn.metrics import precision_recall_curve, roc_curve
import scanpypip.utils as ut

def highly_variable_genes(data, 
    layer=None, n_top_genes=None, 
    min_disp=0.5, max_disp=np.inf, min_mean=0.0125, max_mean=3, 
    span=0.3, n_bins=20, flavor='seurat', subset=False, inplace=True, batch_key=None, PCA_graph=False, PCA_dim = 50, k = 10, n_pcs=40):

    adata = sc.AnnData(data)

    adata.var_names_make_unique() 
    adata.obs_names_make_unique()

    if n_top_genes is not None:
        sc.pp.highly_variable_genes(adata, layer=layer, n_top_genes=n_top_genes,
        span=span, n_bins=n_bins, flavor='seurat_v3', subset=subset, inplace=inplace, batch_key=batch_key)
    else: 
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata,
        layer=layer, n_top_genes=n_top_genes,
        min_disp=min_disp, max_disp=max_disp, min_mean=min_mean, max_mean=max_mean, 
        span=span, n_bins=n_bins, flavor=flavor, subset=subset, inplace=inplace, batch_key=batch_key)
    
    if PCA_graph:
        sc.tl.pca(adata, n_comps=PCA_dim)
        X_pca = adata.obsm["X_pca"]
        sc.pp.neighbors(adata, n_neighbors=k, n_pcs=n_pcs)
        return adata.var.highly_variable, adata, X_pca
    
    return adata.var.highly_variable, adata

def save_arguments(args, now):
    args_strings = re.sub("\'|\"|Namespace|\(|\)", "", str(args)).split(sep=', ')
    args_dict = {item.split('=')[0]: item.split('=')[1] for item in args_strings}

    args_df = pd.DataFrame(args_dict, index=[now]).T
    args_df.to_csv(f"save/log/arguments_{now}.csv")

    return args_df

def plot_label_hist(Y, save=None):
    n, bins, patches = plt.hist(Y, 50, density=True, facecolor='g', alpha=0.75)

    plt.xlabel('Y values')
    plt.ylabel('Probability')
    plt.title('Histogram of target')

    if save is None:
        plt.show()
    else:
        plt.savefig(save)

def plot_roc_curve(test_y, naive_probs, model_probs, title="", path="figures/roc_curve.pdf"):
    fpr, tpr, _ = roc_curve(test_y, naive_probs)
    plt.plot(fpr, tpr, linestyle='--', label='Random')
    fpr, tpr, _ = roc_curve(test_y, model_probs)
    plt.plot(fpr, tpr, marker='.', label='Prediction')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title(title)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
    plt.close()

def plot_pr_curve(test_y, model_probs, selected_label=1, title="", path="figures/prc_curve.pdf"):
    no_skill = len(test_y[test_y == selected_label]) / len(test_y)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Random')
    precision, recall, _ = precision_recall_curve(test_y, model_probs)
    plt.plot(recall, precision, marker='.', label='Prediction')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.title(title)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
    plt.close()

def specific_process(adata, dataname='', **kargs):
    if dataname == 'GSE165318':
        adata = process_165318(adata)
    return adata

def process_165318(adata, **kargs):
    # Load metadata
    meta_data_path = 'data/GSE165318_metadata_singlecell_modified_with_controls_new.csv'  # Update with your actual path
    meta_data = pd.read_csv(meta_data_path, index_col=0)

    # Merging the metadata
    adata.obs = adata.obs.merge(meta_data, left_index=True, right_index=True, how='left')

    return adata

def de_score(adata, clustername, pval=0.05, n=50, method='wilcoxon', score_prefix=None):
    try:
        sc.tl.rank_genes_groups(adata, clustername, method=method, use_raw=True)
    except:
        sc.tl.rank_genes_groups(adata, clustername, method=method, use_raw=False)
    for cluster in set(adata.obs[clustername]):
        df = ut.get_de_dataframe(adata, cluster)
        select_df = df.iloc[:n, :]
        if pval is not None:
            select_df = select_df.loc[df.pvals_adj < pval]
        sc.tl.score_genes(adata, select_df.names, score_name=str(cluster) + '_score')
    return adata

def integrated_gradient_check(net, input, target, adata, n_genes, target_class=1, test_value='expression', save_name='feature_gradients', batch_size=100):
    ig = IntegratedGradients(net)
    attr, delta = ig.attribute(input, target=target_class, return_convergence_delta=True, internal_batch_size=batch_size)
    attr = attr.detach().cpu().numpy()
    adata.var['integrated_gradient_sens_class' + str(target_class)] = attr.mean(axis=0)
    sen_index = (target == 1)
    res_index = (target == 0)
    attr = pd.DataFrame(attr, columns=adata.var.index)
    df_top_genes = adata.var.nlargest(n_genes, 'integrated_gradient_sens_class' + str(target_class), keep='all')
    df_tail_genes = adata.var.nsmallest(n_genes, 'integrated_gradient_sens_class' + str(target_class), keep='all')
    list_topg = df_top_genes.index
    list_tailg = df_tail_genes.index
    top_pvals = []
    tail_pvals = []
    if test_value == 'gradient':
        feature_sens = attr[sen_index]
        feature_rest = attr[res_index]
    else:
        expression_norm = input.detach().cpu().numpy()
        expression_norm = pd.DataFrame(expression_norm, columns=adata.var.index)
        feature_sens = expression_norm[sen_index]
        feature_rest = expression_norm[res_index]
    for g in list_topg:
        f_sens = feature_sens.loc[:, g]
        f_rest = feature_rest.loc[:, g]
        stat, p = mannwhitneyu(f_sens, f_rest)
        top_pvals.append(p)
    for g in list_tailg:
        f_sens = feature_sens.loc[:, g]
        f_rest = feature_rest.loc[:, g]
        stat, p = mannwhitneyu(f_sens, f_rest)
        tail_pvals.append(p)
    df_top_genes['pval'] = top_pvals
    df_tail_genes['pval'] = tail_pvals
    df_top_genes.to_csv('save/results/top_genes_class' + str(target_class) + save_name + '.csv')
    df_tail_genes.to_csv('save/results/top_genes_class' + str(target_class) + save_name + '.csv')
    return adata, attr, df_top_genes, df_tail_genes

def integrated_gradient_differential(net, input, target, adata, n_genes=None, target_class=1, clip='abs', save_name='feature_gradients', ig_pval=0.05, ig_fc=1, method='wilcoxon', batch_size=100):
    ig = IntegratedGradients(net)
    df_results = {}
    attr, delta = ig.attribute(input, target=target_class, return_convergence_delta=True, internal_batch_size=batch_size)
    attr = attr.detach().cpu().numpy()
    if clip == 'positive':
        attr = np.clip(attr, a_min=0, a_max=None)
    elif clip == 'negative':
        attr = abs(np.clip(attr, a_min=None, a_max=0))
    else:
        attr = abs(attr)
    igadata = sc.AnnData(attr)
    igadata.var.index = adata.var.index
    igadata.obs.index = adata.obs.index
    igadata.obs['sensitive'] = target
    igadata.obs['sensitive'] = igadata.obs['sensitive'].astype('category')
    sc.tl.rank_genes_groups(igadata, 'sensitive', method=method, n_genes=n_genes)
    for label in [0, 1]:
        try:
            df_degs = ut.get_de_dataframe(igadata, label)
            df_degs = df_degs.loc[(df_degs.pvals_adj < ig_pval) & (df_degs.logfoldchanges >= ig_fc)]
            df_degs.to_csv('save/results/DIG_class_' + str(target_class) + '_' + str(label) + save_name + '.csv')
            df_results[label] = df_degs
        except:
            logging.warning('Only one class, no two calsses critical genes')
    return adata, igadata, list(df_results[0].names), list(df_results[1].names)

def plot_loss(report, path='figures/loss.pdf', set_ylim=False):
    train_loss = []
    val_loss = []
    epochs = int(len(report) / 2)
    score_dict = {'train': train_loss, 'val': val_loss}
    for phrase in ['train', 'val']:
        for i in range(epochs):
            score_dict[phrase].append(report[(i, phrase)])
    plt.close()
    plt.clf()
    x = np.linspace(0, epochs, epochs)
    plt.plot(x, val_loss, '-g', label='validation loss')
    plt.plot(x, train_loss, ':b', label='training loss')
    plt.legend(['validation loss', 'training loss'], loc='upper left')
    if set_ylim:
        plt.ylim(set_ylim)
    plt.savefig(path)
    plt.close()
    return score_dict

import scanpy as sc

# Provide the correct path to the normalized data file
data_path = "data/GSE165318_normalized.single.cell.transposed.txt"
# Provide the correct path to the metadata file
meta_data_path = "data/GSE165318_metadata_singlecell_modified_with_controls_new.csv"

# Load normalized data
adata = sc.read_text(data_path).transpose()

# Load metadata
meta_data = pd.read_csv(meta_data_path, index_col=0)

# Ensure the metadata is merged into the AnnData object
adata.obs = adata.obs.merge(meta_data, left_index=True, right_index=True, how='left')

# Perform basic preprocessing steps
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var.highly_variable]
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.tl.leiden(adata)
