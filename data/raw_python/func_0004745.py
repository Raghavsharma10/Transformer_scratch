def cannon_normalize(spec_raw):
    """ Normalize according to The Cannon """
    spec = np.array([spec_raw])
    wl = np.arange(0, spec.shape[1])
    w = continuum_normalization.gaussian_weight_matrix(wl, L=50)
    ivar = np.ones(spec.shape)*0.5
    cont = continuum_normalization._find_cont_gaussian_smooth(
            wl, spec, ivar, w)
    norm_flux, norm_ivar = continuum_normalization._cont_norm(
            spec, ivar, cont)
    return norm_flux[0]