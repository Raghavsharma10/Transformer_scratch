def get_linc_rna_genes(
        path_or_buffer,
        remove_duplicates=True,
        **kwargs):
    r"""Get list of all protein-coding genes based on Ensembl GTF file.
    
    Parameters
    ----------
    See :func:`get_genes` function.

    Returns
    -------
    `pandas.DataFrame`
        Table with rows corresponding to protein-coding genes.

    """
    valid_biotypes = set(['lincRNA'])
    
    df = get_genes(path_or_buffer, valid_biotypes,
                   remove_duplicates=remove_duplicates, **kwargs)
    return df