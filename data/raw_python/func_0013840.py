def load_ascii_data(fname, tag):
    """Load data from a self-documenting ASCII SuperMAG file

    Parameters
    ------------
    fname : (str)
        ASCII SuperMAG filename
    tag : (str)
        Denotes type of file to load.  Accepted types are 'indices', 'all',
        'stations', and '' (for just magnetometer measurements).

    Returns
    --------
    data : (pandas.DataFrame)
        Pandas DataFrame
    baseline : (list)
        List of strings denoting the presence of a standard and file-specific
        baselines for each file.  None of not present or not applicable.
        
    """
    import re
    ndata = {"indices":2, "":4, "all":4, "stations":8}
    dkeys = {'stations':list(), '':['IAGA', 'N', 'E', 'Z']}
    data = pds.DataFrame(None)
    baseline = None

    # Ensure that the tag indicates a type of data we know how to load
    if not tag in ndata.keys():
        return data, baseline

    # Read in the text data, processing the header, indices, and
    # magnetometer data (as desired)
    with open(fname, "r") as fopen:
        # Set the processing flags
        hflag = True  # header lines
        pflag = False # parameter line
        dflag = False if tag == "stations" else True  # date line
        snum = 0      # number of magnetometer stations
        ddict = dict()
        date_list = list()

        if tag == "stations":
            dtime = pds.datetime.strptime(fname.split("_")[-1].split(".")[0],
                                          "%Y")

        for fline in fopen.readlines():
            # Cycle past the header
            line_len = len(fline)

            if hflag:
                if pflag:
                    pflag = False # Unset the flag
                    if fline.find("-mlt") > 0:
                        ndata[''] += 2
                        dkeys[''].extend(['MLT', 'MLAT'])
                    if fline.find("-sza") > 0:
                        ndata[''] += 1
                        dkeys[''].append('SZA')
                    if fline.find("-decl") > 0:
                        ndata[''] += 1
                        dkeys[''].append('IGRF_DECL')
                    if tag == "indices" and fline.find("-envelope") < 0:
                        # Indices not included in this file
                        break

                    # Save the baseline information
                    lsplit = fline.split()
                    idelta = lsplit.index('-delta') + 1
                    ibase = lsplit.index('-baseline') + 1
                    isd = lsplit.index('-sd') + 1
                    ist = lsplit.index('-st') + 1
                    iex = lsplit.index('-ex') + 1
                    baseline = " ".join([lsplit[ibase], lsplit[idelta],
                                         lsplit[isd], lsplit[ist], lsplit[iex]])

                if fline.find("Selected parameters:") >= 0:
                    pflag = True
                if fline.count("=") == line_len - 1 and line_len > 2:
                    hflag = False
            else:
                # Load the desired data
                lsplit = [ll for ll in re.split(r'[\t\n]+', fline)
                          if len(ll) > 0]

                if dflag:
                    dflag = False # Unset the date flag
                    dstring = " ".join(lsplit[:6])
                    dtime = pysat.datetime.strptime(dstring,
                                                    "%Y %m %d %H %M %S")
                    snum = int(lsplit[6]) # Set the number of stations

                    # Load the times
                    if tag == "indices":
                        date_list.append(dtime)
                    else:
                        date_list.extend([dtime for i in range(snum)])
                elif len(lsplit) == ndata['indices']:
                    if tag is not '':
                        if lsplit[0] not in ddict.keys():
                            ddict[lsplit[0]] = list()

                        if tag == "indices":
                            ddict[lsplit[0]].append(int(lsplit[1]))
                        else:
                            # This works because indices occur before
                            # magnetometer measurements
                            ddict[lsplit[0]].extend([int(lsplit[1])
                                                     for i in range(snum)])
                else:
                    if tag == "stations" and len(lsplit) >= ndata[tag]:
                        if len(dkeys[tag]) == 0:
                            # Station files include column names and data files
                            # do not.  Read in the column names here
                            for ll in lsplit:
                                ll = re.sub("-", "_", ll)
                                dkeys[tag].append(ll)
                                ddict[ll] = list()
                        else:
                            # Because stations can have multiple operators,
                            # ndata supplies the minimum number of columns
                            date_list.append(dtime)
                            for i,ll in enumerate(lsplit):
                                if i >= 1 and i <= 4:
                                    ddict[dkeys[tag][i]].append(float(ll))
                                elif i == 6:
                                    ddict[dkeys[tag][i]].append(int(ll))
                                elif i < len(dkeys[tag]):
                                    ddict[dkeys[tag][i]].append(ll)
                                else:
                                    ddict[dkeys[tag][-1]][-1] += \
                                                            " {:s}".format(ll)
                    elif len(lsplit) == ndata['']:
                        snum -= 1 # Mark the ingestion of a station
                        if tag != "indices":
                            if len(ddict.keys()) < ndata['']:
                               for kk in dkeys['']:
                                   ddict[kk] = list()
                            for i,kk in enumerate(dkeys['']):
                                if i == 0:
                                    ddict[kk].append(lsplit[i])
                                else:
                                    ddict[kk].append(float(lsplit[i]))

                if tag != "stations" and snum == 0 and len(ddict.items()) >= 2:
                    # The previous value was the last value, prepare for
                    # next block
                    dflag = True

        # Create a data frame for this file
        data = pds.DataFrame(ddict, index=date_list, columns=ddict.keys())

        fopen.close()

    return data, baseline