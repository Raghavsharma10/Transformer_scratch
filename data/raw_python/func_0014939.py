def loadDict(filename):
    """Return the variables pickled pickled into `filename` with `saveVars`
    as a dict."""
    filename = os.path.expanduser(filename)
    if not splitext(filename)[1]: filename += ".bpickle"
    f = None
    try:
        f = open(filename, "rb")
        varH = cPickle.load(f)
    finally:
        if f: f.close()
    return varH