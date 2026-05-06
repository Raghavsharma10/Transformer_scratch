def loadcsv(filename):
    """Load data from CSV file.

    Returns a single dict with column names as keys.
    """
    dataframe = _pd.read_csv(filename)
    data = {}
    for key, value in dataframe.items():
        data[key] = value.values
    return data