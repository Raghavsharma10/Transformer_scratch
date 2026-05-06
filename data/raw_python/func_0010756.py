def read_knmi_dataset(directory):
    """Reads files from a directory and merges the time series

    Please note: For each station, a separate directory must be provided!
    data availability: www.knmi.nl/nederland-nu/klimatologie/uurgegevens

    Args:
        directory: directory including the files

    Returns:
        pandas data frame including time series
    """
    filemask = '%s*.txt' % directory
    filelist = glob.glob(filemask)

    columns_hourly = ['temp', 'precip', 'glob', 'hum', 'wind', 'ssd']
    ts = pd.DataFrame(columns=columns_hourly)

    first_call = True
    for file_i in filelist:
        print(file_i)
        current = read_single_knmi_file(file_i)
        if(first_call):
            ts = current
            first_call = False
        else:
            ts = pd.concat([ts, current])
    return ts