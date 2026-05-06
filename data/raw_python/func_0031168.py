def get_protein_coding_genes(
        path_or_buffer, 
        include_polymorphic_pseudogenes=True,
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
    valid_biotypes = set(['protein_coding'])
    if include_polymorphic_pseudogenes:
        valid_biotypes.add('polymorphic_pseudogene')
    
    df = get_genes(path_or_buffer, valid_biotypes,
                   remove_duplicates=remove_duplicates, **kwargs)
    return df