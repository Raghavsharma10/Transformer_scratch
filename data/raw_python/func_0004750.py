def load_model():
    """ Load the model 

    Parameters
    ----------
    direc: directory with all of the model files
    
    Returns
    -------
    m: model object
    """
    direc = "/home/annaho/TheCannon/code/lamost/mass_age/cn"
    m = model.CannonModel(2)
    m.coeffs = np.load(direc + "/coeffs.npz")['arr_0'][0:3626,:] # no cols
    m.scatters = np.load(direc + "/scatters.npz")['arr_0'][0:3626] # no cols
    m.chisqs = np.load(direc + "/chisqs.npz")['arr_0'][0:3626] # no cols
    m.pivots = np.load(direc + "/pivots.npz")['arr_0']
    return m