def loadhdf(filename, groupname="data", to_dataframe=False):
    """Load all data from top level of HDF5 file--similar to how
    `scipy.io.loadmat` works.
    """
    data = {}
    with _h5py.File(filename, "r") as f:
        for key, value in f[groupname].items():
            data[key] = np.array(value)
    if to_dataframe:
        return _pd.DataFrame(data)
    else:
        return data