import scanpy as sc

def create_count_matrix(
        file_path : str = None
        ) -> sc.AnnData:
    """
    ### Function that transforms 10x h5-file to a count matrix.

    ---
    
    #### Args:
        - file_path (str): Path to a 10x hdf5 file. Default None

    #### Returns: 
        - AnnData: Count matrix from h5-file in file_path with unique variables

    ---

    Annotated data matrix, where observations/cells are named by their barcode and variables/genes by gene name. Stores the following information:

    `~anndata.AnnData.X`
        The data matrix is stored
    `~anndata.AnnData.obs_names`
        Cell names
    `~anndata.AnnData.var_names`
        Gene names
    `~anndata.AnnData.var`\ `['gene_ids']`
        Gene IDs
    `~anndata.AnnData.var`\ `['feature_types']`
        Feature types
    ---

    Written: ronjah@chalmers.se
    """
    if file_path == None:
        raise ValueError('Need a path to h5-file.')
    adata = sc.read_10x_h5(file_path)
    adata.var_names_make_unique()
    return adata


def quality_control(adata: sc.AnnData = None):
    """
    ### Function that adds QC to AnnData object

    ---
    
    #### Args:
        - adata (AnnData): AnnData object to perform quality control on.

    ---
    Written: ronjah@chalmers.se
    """
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, 
                               qc_vars=['mt'], 
                               percent_top=None, 
                               log1p=False, 
                               inplace=True)


def remove_bad_cells(adata: sc.AnnData = None, 
                     max_n_genes: int = 2500, 
                     min_n_genes: int = 200, 
                     mitochondrial_percent: int = 5
                     ) -> sc.AnnData:
    """
    ### Function that removes bad cells

    ---
    
    #### Args:
        - adata (AnnData): AnnData object that has gone through quality control (QC). Default None
        - max_n_genes (int): Remove cells with a gene count over max_n_genes. Default 2500
        - min_n_genes (int): Remove cells with a gene count less than min_n_genes. Default 200
        - mitochondrial_percent (int): Remove cells with a mitochondrial percent above mitochondrial_percent. Default 5

    #### Returns: 
        - AnnData: Where bad cells has been removed.

    ---
    Written: ronjah@chalmers.se
    """
    adata = adata[adata.obs.n_genes_by_counts < max_n_genes, :]
    adata = adata[adata.obs.n_genes_by_counts > min_n_genes, :]
    adata = adata[adata.obs.pct_counts_mt < mitochondrial_percent, :]
    return adata

def normalize_data(
        adata: sc.AnnData,
        target_sum: float = 1e4,
        exclude_highly_expressed: bool = False,
        min_mean: float = 0.0125,
        max_mean: float = 3,
        min_disp: float = 0.5):
    """
    ### Function that normalize the AnnData

    ---
    
    #### Args:
        - adata (AnnData): AnnData object that has gone through quality control (QC).
        - target_sum (float): Float each cell sums up to. Default 1e4
        - exclude_highly_expressed (bool): If true, highly expressed cells are not part of normalization. Default False
        - min_mean (float): Minumum cutoff for means. Default 0.0125
        - max_mean (float): Maximum cutoff for means. Default 3
        - min_disp (float): The threshold for the minimum dispersion a gene must have to be considered highly variable. Default 0.5

    ---
    Written: ronjah@chalmers.se
    """
    sc.pp.normalize_total(adata, 
                          target_sum=target_sum,
                          exclude_highly_expressed=exclude_highly_expressed)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, 
                                min_mean=min_mean, 
                                max_mean=max_mean, 
                                min_disp=min_disp)


def pre_process_data_pipeline(
        file_path: str=None, 
        plots: bool = False,
        max_n_genes: int = 2500,
        min_n_genes: int = 200,
        mitochondrial_percent: int = 5,
        target_sum: float = 1e4,
        exclude_highly_expressed: bool = False,
        min_mean: float = 0.0125,
        max_mean: float = 3,
        min_disp: float = 0.5,
        n_top: int = 20,
        save_format: str = 'png'
        ) -> sc.AnnData:
    """
    ### Function that acts as a pipeline for pre-processing the data.

    ---
    
    #### Args:
        - file_path (str): File path to h5-file, default None
        - plots (bool): True if want to save plots, else false. Default False
        - max_n_genes (int): Remove cells with a gene count over max_n_genes. Default 2500
        - min_n_genes (int): Remove cells with a gene count less than min_n_genes. Default 200
        - mitochondrial_percent (int): Remove cells with a mitochondrial percent above mitochondrial_percent. Default 5
        - target_sum (float): Float each cell sums up to. Default 1e4
        - exclude_highly_expressed (bool): If true, highly expressed cells are not part of normalization. Default False
        - min_mean (float): Minumum cutoff for means. Default 0.0125
        - max_mean (float): Maximum cutoff for means. Default 3
        - min_disp (float): The threshold for the minimum dispersion a gene must have to be considered highly variable. Default 0.5
        - n_top (int): The n_top genes with the highest mean fraction over all cells are plotted as boxplots. Default 20
        - save_format (str): File format for saved plots. Allowed formats: ['png', 'pdf', 'svg']. Default 'png'

    #### Returns: 
        - AnnData: Pre-processed

    ---
    ##### Order of operations:
        1. Create count matrix
        2. Quality control (QC)
        3. Remove bad cells
        4. Normalize data

    ---
    Written: ronjah@chalmers.se
    """
    if file_path == None:
        raise ValueError('Need a path to h5-file.')
    
    
    adata = create_count_matrix(file_path)

    if plots:
        allowed_formats = ['png', 'pdf', 'svg']
        if save_format not in allowed_formats:
            raise ValueError(f"Invalid save_format: '{save_format}'. It must be one of {allowed_formats}")
        
        sc.pl.highest_expr_genes(adata, 
                                 n_top=n_top,
                                 save='.'+ save_format,
                                 show=False)

    quality_control(adata)

    if plots:
        sc.pl.violin(adata, 
                     ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], 
                     jitter=0.4, 
                     multi_panel=True,
                     save='.'+ save_format,
                     show=False)

        sc.pl.scatter(adata, 
                      x='total_counts', 
                      y='pct_counts_mt',
                      save='_pct_counts_mt' +'.'+ save_format,
                      show=False)
        
        sc.pl.scatter(adata, 
                      x='total_counts', 
                      y='n_genes_by_counts',
                      save='_n_genes_by_count' + '.'+ save_format,
                      show=False)

    adata = remove_bad_cells(adata, 
                             max_n_genes=max_n_genes,
                             min_n_genes=min_n_genes,
                             mitochondrial_percent=mitochondrial_percent)
    normalize_data(adata,
                   target_sum=target_sum,
                   exclude_highly_expressed=exclude_highly_expressed,
                   min_mean=min_mean,
                   max_mean=max_mean,
                   min_disp=min_disp)

    if plots:
        sc.pl.highly_variable_genes(adata,
                                    save = '.'+ save_format,
                                    show=False)

    return adata

