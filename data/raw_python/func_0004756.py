def get_err(snr):
    """ Get approximate scatters from SNR
    as determined in the code, snr_test.py
    Order: Teff, logg, MH, CM, NM, alpha """

    quad_terms = np.array(
            [3.11e-3, 1.10e-5, 6.95e-6, 5.05e-6, 4.65e-6, 4.10e-6])
    lin_terms = np.array(
            [-0.869, -2.07e-3, -1.40e-3, -1.03e-3, -1.13e-3, -7.29e-4])
    consts = np.array([104, 0.200, 0.117, 0.114, 0.156, 0.0624])
    err = quad_terms[:,None] * snr**2 + lin_terms[:,None] * snr + consts[:,None]

    # find the minimum of the quadratic function
    min_snr = -lin_terms / (2*quad_terms)
    min_err = quad_terms * min_snr**2 + lin_terms * min_snr + consts
    mask = (snr[:,None] > min_snr).T
    for i in range(0,len(min_err)):
        err[i][mask[i]] = min_err[i]

    return err