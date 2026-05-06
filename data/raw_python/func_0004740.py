def get_model(LAB_DIR):
    """ Cannon model params """
    coeffs = np.load("%s/coeffs.npz" %LAB_DIR)['arr_0']
    scatters = np.load("%s/scatters.npz" %LAB_DIR)['arr_0']
    chisqs = np.load("%s/chisqs.npz" %LAB_DIR)['arr_0']
    pivots = np.load("%s/pivots.npz" %LAB_DIR)['arr_0']
    return coeffs, scatters, chisqs, pivots