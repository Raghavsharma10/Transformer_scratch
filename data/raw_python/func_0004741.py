def get_labels(ids_find):
    """ Labels to make Cannon model spectra """
    a = pyfits.open("%s/lamost_catalog_full.fits" %LAB_DIR)
    data = a[1].data
    a.close()
    id_all = data['lamost_id']
    id_all = np.array(id_all)
    id_all = np.array([val.strip() for val in id_all])
    snr_all = data['cannon_snrg']
    chisq_all = data['cannon_chisq']
    teff = data['cannon_teff']
    logg = data['cannon_logg']
    feh = data['cannon_m_h']
    afe = data['cannon_alpha_m']
    ak = data['cannon_a_k']
    labels = np.vstack((teff,logg,feh,afe,ak))
    choose = np.in1d(id_all, ids_find)
    id_choose = id_all[choose]
    label_choose = labels[:,choose]
    snr_choose = snr_all[choose]
    chisq_choose = chisq_all[choose]
    inds = np.array([np.where(id_choose==val)[0][0] for val in ids_find])
    print(id_choose[inds][100])
    print(ids_find[100])
    return label_choose[:,inds], snr_choose[inds], chisq_choose[inds]