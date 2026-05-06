def _cont_norm_gaussian_smooth(dataset, L):
    """ Continuum normalize by dividing by a Gaussian-weighted smoothed spectrum

    Parameters
    ----------
    dataset: Dataset
        the dataset to continuum normalize
    L: float
        the width of the Gaussian used for weighting

    Returns
    -------
    dataset: Dataset
        updated dataset
    """
    print("Gaussian smoothing the entire dataset...")
    w = gaussian_weight_matrix(dataset.wl, L)

    print("Gaussian smoothing the training set")
    cont = _find_cont_gaussian_smooth(
            dataset.wl, dataset.tr_flux, dataset.tr_ivar, w)
    norm_tr_flux, norm_tr_ivar = _cont_norm(
            dataset.tr_flux, dataset.tr_ivar, cont)
    print("Gaussian smoothing the test set")
    cont = _find_cont_gaussian_smooth(
            dataset.wl, dataset.test_flux, dataset.test_ivar, w)
    norm_test_flux, norm_test_ivar = _cont_norm(
            dataset.test_flux, dataset.test_ivar, cont)
    return norm_tr_flux, norm_tr_ivar, norm_test_flux, norm_test_ivar