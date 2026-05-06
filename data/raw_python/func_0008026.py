def integrate_scanpy(adatas, **kwargs):
    """Integrate a list of `scanpy.api.AnnData`.

    Parameters
    ----------
    adatas : `list` of `scanpy.api.AnnData`
        Data sets to integrate.
    kwargs : `dict`
        See documentation for the `integrate()` method for a full list of
        parameters to use for batch correction.

    Returns
    -------
    integrated
        Returns a list of `np.ndarray` with integrated low-dimensional
        embeddings.
    """
    datasets_dimred, genes = integrate(
        [adata.X for adata in adatas],
        [adata.var_names.values for adata in adatas],
        **kwargs
    )

    return datasets_dimred