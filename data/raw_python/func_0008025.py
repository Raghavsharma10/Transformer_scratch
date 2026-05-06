def correct_scanpy(adatas, **kwargs):
    """Batch correct a list of `scanpy.api.AnnData`.

    Parameters
    ----------
    adatas : `list` of `scanpy.api.AnnData`
        Data sets to integrate and/or correct.
    kwargs : `dict`
        See documentation for the `correct()` method for a full list of
        parameters to use for batch correction.

    Returns
    -------
    corrected
        By default (`return_dimred=False`), returns a list of
        `scanpy.api.AnnData` with batch corrected values in the `.X` field.

    corrected, integrated
        When `return_dimred=False`, returns a two-tuple containing a list of
        `np.ndarray` with integrated low-dimensional embeddings and a list
        of `scanpy.api.AnnData` with batch corrected values in the `.X`
        field.
    """
    if 'return_dimred' in kwargs and kwargs['return_dimred']:
        datasets_dimred, datasets, genes = correct(
            [adata.X for adata in adatas],
            [adata.var_names.values for adata in adatas],
            **kwargs
        )
    else:
        datasets, genes = correct(
            [adata.X for adata in adatas],
            [adata.var_names.values for adata in adatas],
            **kwargs
        )

    new_adatas = []
    for i, adata in enumerate(adatas):
        adata.X = datasets[i]
        new_adatas.append(adata)

    if 'return_dimred' in kwargs and kwargs['return_dimred']:
        return datasets_dimred, new_adatas
    else:
        return new_adatas