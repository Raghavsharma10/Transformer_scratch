def get_labels():
    """ Labels to make Cannon model spectra """
    cannon_teff = data['cannon_teff_2']
    cannon_logg = data['cannon_logg_2']
    cannon_m_h = data['cannon_m_h']
    cannon_alpha_m = data['cannon_alpha_m']
    cannon_a_k = data['cannon_a_k']
    labels = np.vstack(
            (cannon_teff, cannon_logg, cannon_m_h, cannon_alpha_m, cannon_a_k))
    cannon_chisq = data['cannon_chisq']
    np.savez(DATA_DIR + "chisq.npz", labels)
    np.savez(DATA_DIR + "labels.npz", labels)
    snrg = data['cannon_snrg'] # snrg * 3
    np.savez("snr.npz", snrg)
    return labels.T