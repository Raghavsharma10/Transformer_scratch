def read_smet(filename, mode):
    """Reads smet data and returns the data in required dataformat (pd df)

    See https://models.slf.ch/docserver/meteoio/SMET_specifications.pdf
    for further details on the specifications of this file format.

    Parameters
    ----
    filename : SMET file to read
    mode :     "d" for daily and "h" for hourly input

    Returns
    ----
    [header, data]
    header:    header as dict
    data :     data as pd df
    """

    # dictionary
    # based on smet spec V.1.1 and self defined
    # daily data
    dict_d = {'TA': 'tmean',
              'TMAX': 'tmax',   # no spec
              'TMIN': 'tmin',   # no spec
              'PSUM': 'precip',
              'ISWR': 'glob',     # no spec
              'RH': 'hum',
              'VW': 'wind'}

    # hourly data
    dict_h = {'TA': 'temp',
              'PSUM': 'precip',
              'ISWR': 'glob',     # no spec
              'RH': 'hum',
              'VW': 'wind'}

    with open(filename) as f:
        in_header = False
        data_start = None
        header = collections.OrderedDict()

        for line_num, line in enumerate(f):

            if line.strip() == '[HEADER]':
                in_header = True
                continue
            elif line.strip() == '[DATA]':
                data_start = line_num + 1
                break

            if in_header:
                line_split = line.split('=')
                k = line_split[0].strip()
                v = line_split[1].strip()
                header[k] = v

    # get column names
    columns = header['fields'].split()
    multiplier = [float(x) for x in header['units_multiplier'].split()][1:]

    data = pd.read_table(
        filename,
        sep=r'\s+',
        na_values=[-999],
        skiprows=data_start,
        names=columns,
        index_col='timestamp',
        parse_dates=True,
        )

    data = data*multiplier

    del data.index.name

    # rename columns
    if mode == "d":
        data = data.rename(columns=dict_d)
    if mode == "h":
        data = data.rename(columns=dict_h)

    return header, data