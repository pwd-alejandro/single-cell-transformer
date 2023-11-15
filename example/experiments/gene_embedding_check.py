import json

import anndata
import pandas as pd
import numpy as np
import scanpy as sc
import torch
import seaborn as sns
import csv

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from example.scFormer_assets.model_scformer import TransformerModel
from tasks.data_handling import data_pre_processing as dpp

# TODO: Create a function that gives top 9 genes expressed per celltype for a dataset.

random_values = False
normalize_embeddings = True
umap_plot = False
choose_pca = False
pca_plot = False
tsne_plot = False
top_genes = True

filename = f"gene_embeddings_top_genes-test-rkh-pancreas.png"

our_data = dpp.create_count_matrix('../data/pancreas.h5ad',
                                   make_genes_unique=True)


def top_expressed_genes(adata: anndata):
    '''
    Returns with list of top 9 genes where each index corresponds to the index in cell type returned
    '''
    list_top_genes = []
    list_top_genes_names = []
    temp = adata.obs.celltype.reset_index(drop=True)
    cell_types = temp.unique()
    n_categories = len(cell_types)
    adata.var['most_common_in_cell'] = -10 * np.ones(adata.X.shape[1])
    df_top_genes = pd.DataFrame(columns=['gene_index', 'gene_name', 'celltype'])
    for i in range(n_categories):
        cell_name = cell_types[i]
        mask = temp == cell_name

        indices = [i for i, value in enumerate(mask) if value]

        data_subset = adata.X[indices, :]

        column_sums = np.sum(data_subset, axis=0)
        sorted_indices = np.argsort(column_sums)

        top_9_indices = sorted_indices[-9:]
        list_top_genes.append(top_9_indices)
        adata.var['most_common_in_cell'][top_9_indices] = i
        top_9_gene_names = adata.var.index.values[top_9_indices]
        list_top_genes_names.append(top_9_gene_names)
        # d = {f'{cell_name}_indices': top_9_indices, f'{cell_name}_genes': top_9_gene_names}
        # df_top_genes[f'{cell_name}_indices'] = top_9_indices
        # df_top_genes[f'{cell_name}_genes'] = top_9_gene_names
        #df_top_genes[cell_name] = top_9_gene_names
        temp_df = pd.DataFrame(data={'gene_index': top_9_indices.tolist(),
                                     'gene_name': top_9_gene_names.tolist(),
                                     'celltype': [cell_name] * 9})
        df_top_genes = df_top_genes.append(temp_df, ignore_index=True)
    '''
    combined_data = [(arr1, arr2, string) for arr1, arr2, string in zip(list_top_genes, list_top_genes_names, cell_types)]

    # Specify the CSV file path
    csv_file = "gene_indices_cell_type.csv"

    # Write the combined data to the CSV file
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)

        # Write a header if needed
        writer.writerow(["Gene indices", "Cell type"])

        # Write the combined data
        writer.writerows(combined_data)
    breakpoint()
    '''
    return list_top_genes, list_top_genes_names, cell_types, df_top_genes


our_genes = pd.DataFrame({'genes': our_data.var_names.to_numpy()})

with open('../scFormer_assets/vocab.json', "r") as f:
    vocab_dict = json.load(f)

with open('../scFormer_assets/args.json', "r") as f:
    model_configs = json.load(f)

df = pd.DataFrame(data={'tokens': vocab_dict.keys(),
                        'indices': vocab_dict.values()})

df_intersection = df.merge(our_genes, how='inner', left_on='tokens', right_on='genes')

if top_genes:
    top_gene_list_indices, top_gene_list_name, cell_names, df_top_genes = top_expressed_genes(our_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerModel(
    ntoken=len(vocab_dict),
    d_model=model_configs["embsize"],
    nhead=model_configs["nheads"],
    d_hid=model_configs["d_hid"],
    nlayers=model_configs["nlayers"],
    nlayers_cls=model_configs["n_layers_cls"],
    n_cls=1,  # TODO: fix loading this
    vocab=vocab_dict,
    dropout=model_configs["dropout"],
    pad_token=model_configs["pad_token"],
    pad_value=model_configs["pad_value"],
)
model.to(device)

try:
    model.load_state_dict(torch.load('../scFormer_assets/best_model.pt', map_location=device))
except:
    params = model.state_dict()
    for key, value in torch.load('../scFormer_assets/best_model.pt', map_location=device).items():
        # only load params that are in the current model
        if (
                key in model.state_dict()
                and model.state_dict()[key].shape == value.shape
        ):
            params[key] = value
    model.load_state_dict(params)

model.eval()

gene_indices = torch.as_tensor(
    np.array([vocab_dict.get(key) for key in df.tokens.values]))  # df_intersection.genes.values]))
if top_genes:
    df_intersection_top = df.merge(df_top_genes, how='inner', left_on='tokens', right_on='gene_name')
    gene_indices = torch.as_tensor(
        np.array([vocab_dict.get(key) for key in df_intersection_top.gene_name.values]))

breakpoint()

gene_embeddings = model.encoder(gene_indices.to(device)).detach().cpu().numpy()

# random embeddings
if random_values:
    gene_embeddings = torch.rand(size=(gene_embeddings.shape[0], gene_embeddings.shape[1])).detach().cpu().numpy()
    print(gene_embeddings.shape)

# final_data = our_data[:, df_intersection.genes].copy()

# Normalize?
if normalize_embeddings:
    gene_embeddings = gene_embeddings / np.linalg.norm(
        gene_embeddings, axis=1, keepdims=True
    )

# final_data.obsm["X_scFormer_alejandro"] = gene_embeddings
gene_embeddings_anndata = anndata.AnnData(X=gene_embeddings)
if top_genes:
    # TODO: Need the crossover genes, nothing else
    gene_embeddings_anndata.obs['most_common_in_cell'] = our_data.var['most_common_in_cell']
    breakpoint()
df_gene_embeddings = pd.DataFrame(gene_embeddings)

# PCA
if choose_pca:
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(gene_embeddings)
    df_gene_embeddings['pca-one'] = pca_result[:, 0]
    df_gene_embeddings['pca-two'] = pca_result[:, 1]
    df_gene_embeddings['pca-three'] = pca_result[:, 2]

if umap_plot:
    sc.pp.neighbors(gene_embeddings_anndata, use_rep='X')  # final_data, use_rep="X_scFormer_alejandro")
    sc.tl.umap(gene_embeddings_anndata, min_dist=0.3)

    fig = sc.pl.umap(
        gene_embeddings_anndata,
        ncols=2,
        frameon=False,
        return_fig=True,
    )

if top_genes:
    sc.pp.neighbors(gene_embeddings_anndata, use_rep='X')  # final_data, use_rep="X_scFormer_alejandro")
    sc.tl.umap(gene_embeddings_anndata, min_dist=0.3)

    fig = sc.pl.umap(
        gene_embeddings_anndata,
        color=['most_common_in_cell'],
        ncols=1,
        frameon=False,
        return_fig=True,
    )
if pca_plot:
    fig = sns.scatterplot(
        x="pca-one", y="pca-two",
        # hue="y",
        palette=sns.color_palette("hls", 10),
        data=df_gene_embeddings,
        legend="full",
        alpha=0.3
    )
    fig = fig.figure

if tsne_plot:
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df_gene_embeddings)
    df_gene_embeddings['tsne-2d-one'] = tsne_results[:, 0]
    df_gene_embeddings['tsne-2d-two'] = tsne_results[:, 1]

    fig = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        # hue="y",
        palette=sns.color_palette("hls", 10),
        data=df_gene_embeddings,
        legend="full",
        alpha=0.3
    )
    fig = fig.figure

fig.savefig(
    filename,
    bbox_inches="tight",
)

print('done')
