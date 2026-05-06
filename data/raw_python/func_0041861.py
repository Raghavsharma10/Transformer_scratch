def savehdf(filename, datadict, groupname="data", mode="a", metadata=None,
            as_dataframe=False, append=False):
    """Save a dictionary of arrays to file--similar to how `scipy.io.savemat`
    works. If `datadict` is a DataFrame, it will be converted automatically.
    """
    if as_dataframe:
        df = _pd.DataFrame(datadict)
        df.to_hdf(filename, groupname)
    else:
        if isinstance(datadict, _pd.DataFrame):
            datadict = datadict.to_dict("list")
        with _h5py.File(filename, mode) as f:
            for key, value in datadict.items():
                if append:
                    try:
                        f[groupname + "/" + key] = np.append(f[groupname + "/" + key], value)
                    except KeyError:
                        f[groupname + "/" + key] = value
                else:
                    f[groupname + "/" + key] = value
            if metadata:
                for key, value in metadata.items():
                    f[groupname].attrs[key] = value