def pickle_save(thing,fname=None):
    """save something to a pickle file"""
    if fname is None:
        fname=os.path.expanduser("~")+"/%d.pkl"%time.time()
    assert type(fname) is str and os.path.isdir(os.path.dirname(fname))
    pickle.dump(thing, open(fname,"wb"),pickle.HIGHEST_PROTOCOL)
    print("saved",fname)