def save_hdf_metadata(filename, metadata, groupname="data", mode="a"):
    """"Save a dictionary of metadata to a group's attrs."""
    with _h5py.File(filename, mode) as f:
        for key, val in metadata.items():
            f[groupname].attrs[key] = val