def load_dataset(date):
    """ Load the dataset for a single date 
    
    Parameters
    ----------
    date: the date (string) for which to load the data & dataset

    Returns
    -------
    ds: the dataset object
    """
    LAB_DIR = "/home/annaho/TheCannon/data/lamost"
    WL_DIR = "/home/annaho/TheCannon/code/lamost/mass_age/cn"
    SPEC_DIR = "/home/annaho/TheCannon/code/apogee_lamost/xcalib_4labels/output"
    wl = np.load(WL_DIR + "/wl_cols.npz")['arr_0'][0:3626] # no cols
    ds = dataset.Dataset(wl, [], [], [], [], [], [], [])
    test_label = np.load("%s/%s_all_cannon_labels.npz" %(LAB_DIR,date))['arr_0']
    ds.test_label_vals = test_label
    a = np.load("%s/%s_norm.npz" %(SPEC_DIR,date))
    ds.test_flux = a['arr_0']
    ds.test_ivar = a['arr_1']
    ds.test_ID = np.load("%s/%s_ids.npz" %(SPEC_DIR,date))['arr_0']
    return ds