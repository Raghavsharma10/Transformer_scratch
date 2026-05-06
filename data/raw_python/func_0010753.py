def read_dwd(filename, metadata, mode="d", skip_last=True):
    """Reads dwd (German Weather Service) data and returns the data in required
    dataformat (pd df)

    Parameters
    ----
    filename : DWD file to read (full path) / list of hourly files (RR+TU+FF) 
    metadata : corresponding DWD metadata file to read
    mode :    "d" for daily and "h" for hourly input
    skip_last : boolen, skips last line due to file format

    Returns
    ----
    [header, data]
    header:    header as dict
    data :     data as pd df
    """

    def read_single_dwd(filename, metadata, mode, skip_last):
        # Param names {'DWD':'dissag_def'}
        dict_d = {'LUFTTEMPERATUR': 'tmean',
                  'LUFTTEMPERATUR_MINIMUM': 'tmin',   # no spec
                  'LUFTTEMPERATUR_MAXIMUM': 'tmax',   # no spec
                  'NIEDERSCHLAGSHOEHE': 'precip',
                  'GLOBAL_KW_J': 'glob',     # no spec
                  'REL_FEUCHTE': 'hum',
                  'WINDGESCHWINDIGKEIT': 'wind',
                  'SONNENSCHEINDAUER': 'sun_h'}

        # ---read meta------------------
        meta = pd.read_csv(
            metadata,
            sep=';'
            )

        # remove whitespace from header columns
        meta.rename(columns=lambda x: x.strip(), inplace=True)

        header = {"Stations_id": meta.Stations_id[meta.last_valid_index()],
                  "Stationsname": meta.Stationsname[meta.last_valid_index()],
                  # workaround for colnames with . (Geogr.Breite)
                  "Breite": meta.iloc[meta.last_valid_index(), 2],  # DezDeg
                  "Laenge": meta.iloc[meta.last_valid_index(), 3]   # DezDeg
                  }

        # ---read data------------------
        if skip_last is not None:
            num_lines = sum(1 for line in open(filename))
            skip_last = [num_lines-1]

        # hourly data must be parsed by custom definition
        if mode == "d":
            data = pd.read_csv(
                filename,
                sep=';',
                na_values='-999',
                index_col=' MESS_DATUM',
                parse_dates=True,
                skiprows=skip_last
                )

        # hourly data must be parsed by custom definition
        if mode == "h":
            def date_parser(date_time):
                hour = date_time[8:10]
                day = date_time[6:8]
                month = date_time[4:6]
                year = date_time[0:4]
                minute = '00'
                sec = '00'
                return pd.Timestamp('%s-%s-%s %s:%s:%s' % (year, month, day, hour, minute, sec))

            data = pd.read_csv(
                filename,
                sep=';',
                na_values='-999',
                index_col=' MESS_DATUM',
                date_parser=date_parser,
                skiprows=skip_last
                )

        # remove whitespace from header columns
        data.rename(columns=lambda x: x.strip(), inplace=True)

        # rename to dissag definition
        data = data.rename(columns=dict_d)
        # get colums which are not defined
        drop = [col for col in data.columns if col not in dict_d.values()]
        # delete columns
        data = data.drop(drop, axis=1)

        # convert temperatures to Kelvin (+273.15)
        if 'tmin' in data.columns:
            data["tmin"] = data["tmin"] + 273.15
        if 'tmax' in data.columns:
            data["tmax"] = data["tmax"] + 273.15
        if 'tmean' in data.columns:
            data["tmean"] = data["tmean"] + 273.15
        if 'temp' in data.columns:
            data["temp"] = data["temp"] + 273.15

        return header, data

    if type(filename) == list:
        i = 1
        for file in filename:
            header, data_h = read_single_dwd(file, metadata, mode, skip_last)

            if i == 1:
                data = data_h
            else:
                data = data.join(data_h, how='outer')
            i += 1

    else:
        header, data = read_single_dwd(filename, metadata, mode, skip_last)

    return header, data