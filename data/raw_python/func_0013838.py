def load(fnames, tag='', sat_id=None):
    """ Load the SuperMAG files

    Parameters
    -----------
    fnames : (list)
        List of filenames
    tag : (str or NoneType)
        Denotes type of file to load.  Accepted types are 'indices', 'all',
        'stations', and '' (for just magnetometer measurements). (default='')
    sat_id : (str or NoneType)
        Satellite ID for constellations, not used. (default=None)

    Returns
    --------
    data : (pandas.DataFrame)
        Object containing satellite data
    meta : (pysat.Meta)
        Object containing metadata such as column names and units
        
    """

    # Ensure that there are files to load
    if len(fnames) <= 0 :
        return pysat.DataFrame(None), pysat.Meta(None)

    # Ensure that the files are in a list
    if isinstance(fnames, str):
        fnames = [fnames]

    # Initialise the output data
    data = pds.DataFrame()
    baseline = list()

    # Cycle through the files
    for fname in fnames:
        fname = fname[:-11] # Remove date index from end of filename
        file_type = path.splitext(fname)[1].lower()

        # Open and load the files for each file type
        if file_type == ".csv":
            if tag != "indices":
                temp = load_csv_data(fname, tag)
        else:
            temp, bline = load_ascii_data(fname, tag)

            if bline is not None:
                baseline.append(bline)

        # Save the loaded data in the output data structure
        if len(temp.columns) > 0:
            data = pds.concat([data, temp], axis=0)
        del temp

    # If data was loaded, update the meta data
    if len(data.columns) > 0:
        meta = pysat.Meta()
        for cc in data.columns:
            meta[cc] = update_smag_metadata(cc)

        meta.info = {'baseline':format_baseline_list(baseline)}
    else:
        meta = pysat.Meta(None)

    return data, meta