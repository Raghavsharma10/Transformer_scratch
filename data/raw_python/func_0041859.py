def savecsv(filename, datadict, mode="w"):
    """Save a dictionary of data to CSV."""
    if mode == "a" :
        header = False
    else:
        header = True
    with open(filename, mode) as f:
        _pd.DataFrame(datadict).to_csv(f, index=False, header=header)