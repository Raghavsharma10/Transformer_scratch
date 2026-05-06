def draw_spectra(md, ds):
    """ Generate best-fit spectra for all the test objects  

    Parameters
    ----------
    md: model
        The Cannon spectral model

    ds: Dataset 
        Dataset object

    Returns
    -------
    best_fluxes: ndarray 
        The best-fit test fluxes

    best_ivars:
        The best-fit test inverse variances
    """
    coeffs_all, covs, scatters, red_chisqs, pivots, label_vector = model.model
    nstars = len(dataset.test_SNR)
    cannon_flux = np.zeros(dataset.test_flux.shape)
    cannon_ivar = np.zeros(dataset.test_ivar.shape)
    for i in range(nstars):
        x = label_vector[:,i,:]
        spec_fit = np.einsum('ij, ij->i', x, coeffs_all)
        cannon_flux[i,:] = spec_fit
        bad = dataset.test_ivar[i,:] == SMALL**2
        cannon_ivar[i,:][~bad] = 1. / scatters[~bad] ** 2
    return cannon_flux, cannon_ivar