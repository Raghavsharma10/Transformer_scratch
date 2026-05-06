def dictToH5(h5, d, link_copy=False):
    """ Save a dictionary into an hdf5 file
        h5py is not capable of handling dictionaries natively"""
    global _array_cache
    _array_cache = dict()
    h5 = h5py.File(h5, mode="w")
    dictToH5Group(d, h5["/"], link_copy=link_copy)
    h5.close()
    _array_cache = dict();