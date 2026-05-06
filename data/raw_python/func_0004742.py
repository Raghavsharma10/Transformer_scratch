def get_normed_spectra():
    """ Spectra to compare with models """
    wl = np.load("%s/wl.npz" %LAB_DIR)['arr_0']
    filenames = np.array(
            [SPEC_DIR + "/Spectra" + "/" + val for val in lamost_id])
    grid, fluxes, ivars, npix, SNRs = lamost.load_spectra(
            lamost_id, input_grid=wl)
    ds = dataset.Dataset(
            wl, lamost_id, fluxes, ivars, [1], 
            lamost_id[0:2], fluxes[0:2], ivars[0:2])
    ds.continuum_normalize_gaussian_smoothing(L=50)
    np.savez(SPEC_DIR + "/" + "norm_flux.npz", ds.tr_flux)
    np.savez(SPEC_DIR + "/" + "norm_ivar.npz", ds.tr_ivar)
    return ds.tr_flux, ds.tr_ivar