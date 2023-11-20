import scanpy as sc
import pandas as pd


def classify_cells(pre_processed_adata: sc.AnnData = None,
                   class_df: pd.DataFrame = None,
                   save_path: str='classified_temp.h5'
) -> sc.AnnData:
    """
    ### Function that classifies cells and saves the result as a h5-file.

    ---

    #### Args:
        - pre_processed_adata (AnnData): AnnData object that has gone through pre-processing. Default None
        - class_df (DataFrame): DataFrame with categories 'genes' and 'set'. Default None
        - save_path (PathLike): Save path for h5-file, include file name. Default 'classified_temp.h5'

    #### Returns:
        - AnnData: With class scores and classification

    ---
    Written: ronjah@chalmers.se
    """

    adata = pre_processed_adata.copy()
    categories = class_df['set'].unique().tolist()

    score_categories(pre_processed_adata=adata,
                     unique_categories_list=categories,
                     class_df=class_df)

    annotate_categories(scored_adata=adata,
                        unique_categories_list=categories)

    # Save the AnnData object to an h5 file
    adata.write_h5ad(save_path)
    return adata


def score_categories(pre_processed_adata: sc.AnnData = None,
                     unique_categories_list: list = [],
                     class_df: pd.DataFrame = ''
                     ):
    for suva_class in unique_categories_list:
        gene_list = class_df.loc[class_df['set'] == suva_class, 'genes'].tolist()
        sc.tl.score_genes(pre_processed_adata, score_name=suva_class, gene_list=gene_list)
        sc.tl.score_genes(pre_processed_adata, score_name=suva_class, gene_list=gene_list)


def annotate_categories(scored_adata: sc.AnnData = None,
                        unique_categories_list: list = []
                        ):
    # Calculate max value in each cell
    max_class_column = scored_adata.obs[unique_categories_list].idxmax(axis=1)

    # Annotate the result in column 'suva_class'
    scored_adata.obs['suva_class'] = max_class_column
