def save(fname, d, link_copy=True,raiseError=False):
    """ link_copy is used by hdf5 saving only, it allows to creat link of identical arrays (saving space) """
    # make sure the object is dict (recursively) this allows reading it
    # without the DataStorage module
    fname = pathlib.Path(fname)
    d = toDict(d, recursive=True)
    d['filename'] = str(fname)
    extension = fname.suffix
    log.info("Saving storage file %s" % fname)
    try:
        if extension == ".npz":
            return dictToNpz(fname, d)
        elif extension == ".h5":
            return dictToH5(fname, d, link_copy=link_copy)
        elif extension == ".npy":
            return dictToNpy(fname, d)
        else:
            raise ValueError(
                "Extension must be h5, npy or npz, it was %s" % extension)
    except Exception as e:
        log.exception("Could not save %s" % fname)
        if raiseError: raise  e