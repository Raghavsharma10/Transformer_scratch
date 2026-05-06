def load_hdf_metadata(filename, groupname="data"):
    """"Load attrs of the desired group into a dictionary."""
    with _h5py.File(filename, "r") as f:
        data = dict(f[groupname].attrs)
    return data