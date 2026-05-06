def get_residuals(ds, m):
    """ Using the dataset and model object, calculate the residuals and return

    Parameters
    ----------
    ds: dataset object
    m: model object
    Return
    ------
    residuals: array of residuals, spec minus model spec
    """
    model_spectra = get_model_spectra(ds, m)
    resid = ds.test_flux - model_spectra
    return resid