def make_full_ivar():
    """ take the scatters and skylines and make final ivars """

    # skylines come as an ivar
    # don't use them for now, because I don't really trust them...
    # skylines = np.load("%s/skylines.npz" %DATA_DIR)['arr_0']

    ref_flux = np.load("%s/ref_flux_all.npz" %DATA_DIR)['arr_0']
    ref_scat = np.load("%s/ref_spec_scat_all.npz" %DATA_DIR)['arr_0']
    test_flux = np.load("%s/test_flux.npz" %DATA_DIR)['arr_0']
    test_scat = np.load("%s/test_spec_scat.npz" %DATA_DIR)['arr_0']
    ref_ivar = np.ones(ref_flux.shape) / ref_scat[:,None]**2
    test_ivar = np.ones(test_flux.shape) / test_scat[:,None]**2

    # ref_ivar = (ref_ivar_temp * skylines[None,:]) / (ref_ivar_temp + skylines)
    # test_ivar = (test_ivar_temp * skylines[None,:]) / (test_ivar_temp + skylines)

    ref_bad = np.logical_or(ref_flux <= 0, ref_flux > 1.1)
    test_bad = np.logical_or(test_flux <= 0, test_flux > 1.1)
    SMALL = 1.0 / 1000000000.0
    ref_ivar[ref_bad] = SMALL
    test_ivar[test_bad] = SMALL
    np.savez("%s/ref_ivar_corr.npz" %DATA_DIR, ref_ivar)
    np.savez("%s/test_ivar_corr.npz" %DATA_DIR, test_ivar)