def load(fnames, tag=None, sat_id=None):
    """Load CHAMP STAR files

    Parameters
    ------------
    fnames : (pandas.Series)
        Series of filenames
    tag : (str or NoneType)
        tag or None (default=None)
    sat_id : (str or NoneType)
        satellite id or None (default=None)

    Returns
    ---------
    data : (pandas.DataFrame)
        Object containing satellite data
    meta : (pysat.Meta)
        Object containing metadata such as column names and units
    """
    import re
    if len(fnames) <= 0:
        return pysat.DataFrame(None), pysat.Meta(None)

    if isinstance(fnames, str):
        fnames = [fnames]

    # Define the CHAMP STAR data types by column
    champ_labels = { 'Two-digit Year (years)' : "year",
                     'Day of the Year (days)' : "doy",
                     'Second of the Day (GPS time,sec)' : "sod",
                     'Center Latitude of 3-degree Bin (deg)' : "bin_lat",
                     'Satellite Geodetic Latitude (deg)' : "sat_glat",
                     'Satellite Longitude (deg)' : "sat_lon",
                     'Satellite Height (km)' : "sat_h",
                     'Satellite Local Time (hours)' : "sat_lt",
                     'Satellite Quasi-Dipole Latitude (deg)' : "sat_qdlat",
                     'Satellite Magnetic Longitude (deg)' : "sat_mlon",
                     'Satellite Magnetic Local Time (hours)' : "sat_mlt",
                     'Neutral Density (kg/m^3)' : "ndens",
                     'Neutral Density Normalized to 400km using NRLMSISe00' :
                     "ndens400",
                     'Neutral Density Normalized to 410km using NRLMSISe00' :
                     "ndens410",
                     'NRLMSISe00 Neutral Density at Satellite Height' :
                     "nrlmsis_ndens",
                     'Uncertainty in Neutral Density (kg/m^3)' : "ndens_err",
                     'Number of Data Points in Current Averaging Bin' :
                     "npnts", \
    'Number of Points in Current Averaging Bin that Required Interpolation' :
                     "npnts_interp", \
    'Average Coefficient of Drag Used in Current Averaging Bin' :
                     "avg_drag_coeff", }

    champ_dtypes = { 'year' : np.int32, 'doy' : np.int32, 'sod' : float,
                     'bin_lat' : float, 'sat_glat' : float, 'sat_lon' : float,
                     'sat_h' : float, 'sat_lt' : float, 'sat_qdlat' : float,
                     'sat_mlon' : float, 'sat_mlt' : float, 'ndens' : float,
                     'ndens400' : float, 'ndens410' : float,
                     'nrlmsis_ndens' : float, 'ndens_err' : float,
                     'npnts' : int, 'npnts_interp' : float,
                     'avg_drag_coeff' : float, }

    champ_units = { 'year' : "2-digit years", 'doy' : "day of year",
                    'sod' : "seconds of day", 'bin_lat' : "degrees",
                    'sat_glat' : "degrees", 'sat_lon' : "degrees",
                    'sat_h' : "km", 'sat_lt' : "hours", 'sat_qdlat' : "degrees",
                    'sat_mlon' : "degrees", 'sat_mlt' : "hours",
                    'ndens' : "km m^{-3}", 'ndens400' : "km m^{-3}",
                    'ndens410' : "km m^{-3}", 'nrlmsis_ndens' : "km m^{-3}",
                    'ndens_err' : "km m^{-3}", 'npnts' : "number",
                    'npnts_interp' : "number", 'avg_drag_coeff' : "unitless", }

    # Define the routine needed to create datetime object from the
    # CHAMP time (YY DDD SSSSS)
    def parse_champdate(y, d, s):
        '''parse CHAMP date string (YY DDD SSSSS) into a datetime object
        '''
        import datetime as dt

        t = dt.datetime.strptime("{:02d} {:03d}".format(int(y),int(d)), "%y %j")
        fsec = float(s)
        isec = np.floor(fsec)
        microsec = (fsec - isec) * 1.0e6
        t += dt.timedelta(seconds=isec, microseconds=microsec)
        return(t)

    # The header is formatted differently from the rest of the file, read it in
    # first to obtain the necessary meta data
    f = open(fnames[0], "r")
    vdata = re.split( ";|\n", f.readline())
    hdata = re.split( ";|\n", f.readline())
    f.close()
    try:
        hdata.pop(hdata.index(''))
    except:
        pass

    # If there are files, read in the data
    data = pds.read_csv(fnames[0], delim_whitespace=True, skiprows=2,
                        header=None, names=[champ_labels[h] for h in hdata],
                        keep_date_col=True, index_col='datetime',
                        parse_dates={'datetime': [0,1,2]},
                        date_parser=parse_champdate)

    # Initialize the meta data
    meta = pysat.Meta()

    # Because the native dtype declaration interferred with datetime indexing,
    # define the data types here.  Also set the meta data
    for h in hdata:
        col = champ_labels[h]
        data[col].astype(champ_dtypes[col])
        meta[col] = {"units":champ_units[col], "long_name":h}

    # Return data frame and metadata object
    return data, meta