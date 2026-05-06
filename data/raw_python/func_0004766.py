def xvalidate():
    """ Train a model, leaving out a group corresponding
    to a random integer from 0 to 7, e.g. leave out 0. 
    Test on the remaining 1/8 of the sample. """

    print("Loading data")
    groups = np.load("ref_groups.npz")['arr_0']
    ref_label = np.load("%s/ref_label.npz" %direc_ref)['arr_0']
    ref_id = np.load("%s/ref_id.npz" %direc_ref)['arr_0']
    ref_flux = np.load("%s/ref_flux.npz" %direc_ref)['arr_0']
    ref_ivar = np.load("%s/ref_ivar.npz" %direc_ref)['arr_0']
    wl = np.load("%s/wl.npz" %direc_ref)['arr_0']

    num_models = 8

    for ii in np.arange(num_models):
        print("Leaving out group %s" %ii)
        train_on = groups != ii
        test_on = groups == ii

        tr_label = ref_label[train_on]
        tr_id = ref_id[train_on]
        tr_flux = ref_flux[train_on]
        tr_ivar = ref_ivar[train_on]
        print("Training on %s objects" %len(tr_id))
        test_label = ref_label[test_on]
        test_id = ref_id[test_on]
        test_flux = ref_flux[test_on]
        test_ivar = ref_ivar[test_on]
        print("Testing on %s objects" %len(test_id))

        print("Loading dataset...")
        ds = dataset.Dataset(
                wl, tr_id, tr_flux, tr_ivar, tr_label, 
                test_id, test_flux, test_ivar)
        ds.set_label_names(
                ['T_{eff}', '\log g', '[M/H]', '[\\alpha/Fe]', 'AKWISE'])
        fig = ds.diagnostics_SNR()
        plt.savefig("ex%s_SNR.png" %ii)
        fig = ds.diagnostics_ref_labels()
        plt.savefig("ex%s_ref_label_triangle.png" %ii)
        np.savez("ex%s_tr_snr.npz" %ii, ds.tr_SNR)

        # train a model
        m = train(ds, ii)

        # test step
        ds.tr_label = test_label # to compare the results
        test(ds, m, ii)