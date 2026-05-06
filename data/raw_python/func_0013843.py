def download(date_array, tag, sat_id='', data_path=None, user=None,
             password=None, baseline='all', delta='none', options='all',
             file_fmt='ascii'):
    """Routine to download SuperMAG data

    Parameters
    -----------
    date_array : np.array
        Array of datetime objects
    tag : string
        String denoting the type of file to load, accepted values are 'indices',
        'all', 'stations', and '' (for only magnetometer data)
    sat_id : string
        Not used (default='')
    data_path : string or NoneType
        Data path to save downloaded files to (default=None)
    user : string or NoneType
        SuperMAG requires user registration (default=None)
    password : string or NoneType
        Not used; SuperMAG does not require a password (default=None)
    file_fmt : string
        File format options: 'ascii' and 'csv'. (default='ascii')
    baseline : string
        Baseline to remove from magnetometer data.  Options are 'all', 'yearly',
        and 'none'. (default='all')
    delta : string
        Remove a value from the magnetometer data.  Options are 'none', 'start',
        and 'median'.  (default='none')
    options : string or NoneType
        Additional parameter options for magnetometer data.  Includes 'mlt'
        (MLat and MLT), 'decl' (IGRF declination), 'sza' (Solar Zenith Angle),
        'all', and None. (default='all')

    Returns
    -------
    
    """
    import sys
    import requests
    
    global platform, name

    max_stations = 470

    if user is None:
        raise ValueError('SuperMAG requires user registration')

    remoteaccess = {'method':'http', 'host':'supermag.jhuapl.edu',
                    'path':'mag/lib/services', 'user':'user={:s}'.format(user),
                    'service':'service=', 'options':'options='}
    remotefmt = "{method}://{host}/{path}/??{user}&{service}&{filefmt}&{start}"

    # Set the tag information
    if tag == "indices":
        tag = "all"

    if tag != "stations":
        remotefmt += "&{interval}&{stations}&{delta}&{baseline}&{options}"

    # Determine whether station or magnetometer data is requested
    remoteaccess['service'] += tag if tag == "stations" else "mag"

    # Add request for file type
    file_fmt = file_fmt.lower()
    if not file_fmt in ['ascii', 'csv']:
        estr = "unknown file format [{:s}], using 'ascii'".format(file_fmt)
        print("WARNING: {:s}".format(estr))
        file_fmt = 'ascii'
    remoteaccess['filefmt'] = 'fmt={:s}'.format(file_fmt)

    # If indices are requested, add them now.
    if not tag in [None, 'stations']:
        remoteaccess['options'] += "+envelope"

    # Add other download options (for non-station files)
    if tag != "stations":
        if options is not None:
            options = options.lower()
            if options is 'all':
                remoteaccess['options'] += "+mlt+sza+decl"
            else:
                remoteaccess['options'] += "+{:s}".format(options)

        # Add requests for baseline substraction
        baseline = baseline.lower()
        if not baseline in ['all', 'yearly', 'none']:
            estr = "unknown baseline [{:s}], using 'all'".format(baseline)
            print("WARNING: {:s}".format(estr))
            baseline = 'all'
        remoteaccess['baseline'] = "baseline={:s}".format(baseline)

        delta = delta.lower()
        if not delta in ['none', 'median', 'start']:
            estr = "unknown delta [{:s}], using 'none'".format(delta)
            print("WARNING: {:s}".format(estr))
            delta = 'none'
        remoteaccess['delta'] = 'delta={:s}'.format(delta)

        # Set the time information and format
        remoteaccess['interval'] = "interval=23:59"
        sfmt = "%Y-%m-%dT00:00:00.000"
        tag_str = "_" if tag is None else "_all_" 
        ffmt = "{:s}_{:s}{:s}%Y%m%d.{:s}".format(platform, name, tag_str,
                                                 "txt" if file_fmt == "ascii"
                                                 else file_fmt)
        start_str = "start="
    else:
        # Set the time format
        sfmt = "%Y"
        ffmt = "{:s}_{:s}_{:s}_%Y.{:s}".format(platform, name, tag,
                                               "txt" if file_fmt == "ascii"
                                               else file_fmt)
        start_str = "year="

    # Cycle through all of the dates, formatting them to achieve a unique set
    # of times to download data
    date_fmts = list(set([dd.strftime(sfmt) for dd in date_array]))

    # Now that the unique dates are known, construct the file names
    name_fmts = [None for dd in date_fmts]
    for dd in date_array:
        i = date_fmts.index(dd.strftime(sfmt))
        name_fmts[i] = dd.strftime(ffmt)

    if None in name_fmts:
        raise ValueError("unable to construct all unique file names")

    # Cycle through all of the unique dates.  Stations lists are yearly and
    # magnetometer data is daily
    station_year = None
    istr = 'SuperMAG {:s}'.format(tag if tag == "stations" else "data")
    for i,date in enumerate(date_fmts):
        print("Downloading {:s} for {:s}".format(istr, date.split("T")[0]))
        sys.stdout.flush()
        nreq = 1

        # Add the start time and download period to query
        remoteaccess['start'] = "{:s}{:s}".format(start_str, date)
        if tag != "stations":
            # Station lists are for each year, see if this year is loaded
            current_date = pds.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.000")

            if current_date.year != station_year:
                # Get all of the stations for this time
                smag_stat = pysat.Instrument(platform=platform, name=name,
                                             tag='stations')
                # try to load data
                smag_stat.load(date=current_date)
                if smag_stat.empty:
                    # no data
                    etime = current_date + pds.DateOffset(days=1)
                    smag_stat.download(start=current_date, stop=etime,
                                       user=user, password=password,
                                       file_fmt=file_fmt)
                    smag_stat.load(date=current_date)
                    if smag_stat.empty:
                        # no data
                        estr = "unable to format station query for "
                        estr += "[{:d}]".format(current_date.year)
                        raise ValueError(estr)

                # Format a string of the station names
                if smag_stat.data.IAGA.shape[0] > max_stations:
                    station_year = current_date.year
                    nreq = int(np.ceil(smag_stat.data.IAGA.shape[0] /
                                       float(max_stations)))

        out = list()
        for ireq in range(nreq):
            if tag != "stations":
                if station_year is None:
                    raise RuntimeError("unable to load station data")

                stat_str = ",".join(smag_stat.data.IAGA[ireq*max_stations:
                                                        (ireq+1)*max_stations])
                remoteaccess['stations'] = "stations={:s}".format(stat_str)

            # Format the query
            url = remotefmt.format(**remoteaccess)

            # Set up a request
            try:
                # print (url)
                result = requests.post(url)
                result.encoding = 'ISO-8859-1'
                # handle strings differently for python 2/3
                if sys.version_info.major == 2:
                    out.append(str(result.text.encode('ascii', 'replace')))
                else:
                    out.append(result.text)
            except:
                raise RuntimeError("unable to connect to [{:s}]".format(url))

            # Test the result
            if "requested URL was rejected" in out[-1]:
                estr = "Requested url was rejected:\n{:s}".format(url)
                raise RuntimeError(estr)

        # Build the output file name
        if tag is '':
            fname = path.join(data_path, name_fmts[i])
        else:
            fname = path.join(data_path, name_fmts[i])

        # If more than one data pass was needed, append the files
        if len(out) > 1:
            out_data = append_data(out, file_fmt, tag)
        else:
            out_data = out[0]

        # Save the file data
        with open(fname, "w") as local_file:
            local_file.write(out_data)
            local_file.close()
            del out_data

    return