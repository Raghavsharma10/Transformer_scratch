def get_model_spec_ting(atomic_number):
    """ 
    X_u_template[0:2] are teff, logg, vturb in km/s
    X_u_template[:,3] -> onward, put atomic number 
    atomic_number is 6 for C, 7 for N
    """
    DATA_DIR = "/Users/annaho/Data/LAMOST/Mass_And_Age"
    temp = np.load("%s/X_u_template_KGh_res=1800.npz" %DATA_DIR)
    X_u_template = temp["X_u_template"]
    wl = temp["wavelength"]
    grad_spec = X_u_template[:,atomic_number]
    return wl, grad_spec