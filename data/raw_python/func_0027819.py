def pickle_load(fname):
    """return the contents of a pickle file"""
    assert type(fname) is str and os.path.exists(fname)
    print("loaded",fname)
    return pickle.load(open(fname,"rb"))