def load_data():
    data_dir = "/Users/annaho/Data/AAOmega"
    out_dir = "%s/%s" %(data_dir, "Run_13_July")

    """ Use all the above functions to set data up for The Cannon """
    ff, wl, tr_flux, tr_ivar = load_ref_spectra()

    """ pick one that doesn't have extra dead pixels """
    skylines = tr_ivar[4,:] # should be the same across all obj
    np.savez("%s/skylines.npz" %out_dir, skylines)

    contmask = np.load("%s/contmask_regions.npz" %data_dir)['arr_0']
    scatter = estimate_noise(tr_flux, contmask)
    ids, labels = load_labels()
    
    # Select the objects in the catalog corresponding to the files
    inds = []
    ff_short = []
    for fname in ff:
        val = fname.split("/")[-1]
        short = (val.split('.')[0] + '.' + val.split('.')[1])
        ff_short.append(short)
        if short in ids:
            ind = np.where(ids==short)[0][0]
            inds.append(ind)

    # choose the labels
    tr_id = ids[inds]
    tr_label = labels[inds]

    # find the corresponding spectra
    ff_short = np.array(ff_short)
    inds = np.array([np.where(ff_short==val)[0][0] for val in tr_id])
    tr_flux_choose = tr_flux[inds]
    tr_ivar_choose = tr_ivar[inds]
    scatter_choose = scatter[inds]
    np.savez("%s/wl.npz" %out_dir, wl)
    np.savez("%s/ref_id_all.npz" %out_dir, tr_id)
    np.savez("%s/ref_flux_all.npz" %out_dir, tr_flux_choose)
    np.savez("%s/ref_ivar_all.npz" %out_dir, tr_ivar_choose)
    np.savez("%s/ref_label_all.npz" %out_dir, tr_label)
    np.savez("%s/ref_spec_scat_all.npz" %out_dir, scatter_choose)

    # now, the test spectra
    test_id, test_flux = load_test_spectra()
    scatter = estimate_noise(test_flux, contmask) 
    np.savez("%s/test_id.npz" %out_dir, test_id)
    np.savez("%s/test_flux.npz" %out_dir, test_flux)
    np.savez("%s/test_spec_scat.npz" %out_dir, scatter)