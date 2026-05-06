def read_cumulative_iss_index():
    "Read in the whole cumulative index and return dataframe."
    indexdir = get_index_dir()

    path = indexdir / "COISS_2999_index.hdf"
    try:
        df = pd.read_hdf(path, "df")
    except FileNotFoundError:
        path = indexdir / "cumindex.hdf"
        df = pd.read_hdf(path, "df")
    # replace PDS Nan values (-1e32) with real NaNs
    df = df.replace(-1.000000e32, np.nan)
    return df.replace(-999.0, np.nan)