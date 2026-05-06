def load_csv_data(fname, tag):
    """Load data from a comma separated SuperMAG file

    Parameters
    ------------
    fname : (str)
        CSV SuperMAG file name
    tag : (str)
        Denotes type of file to load.  Accepted types are 'indices', 'all',
        'stations', and '' (for just magnetometer measurements).

    Returns
    --------
    data : (pandas.DataFrame)
        Pandas DataFrame
        
    """
    import re

    if tag == "stations":
        # Because there may be multiple operators, the default pandas reader
        # cannot be used.
        ddict = dict()
        dkeys = list()
        date_list = list()

        # Open and read the file
        with open(fname, "r") as fopen:
            dtime = pds.datetime.strptime(fname.split("_")[-1].split(".")[0],
                                          "%Y")

            for fline in fopen.readlines():
                sline = [ll for ll in re.split(r'[,\n]+', fline) if len(ll) > 0]

                if len(ddict.items()) == 0:
                    for kk in sline:
                        kk = re.sub("-", "_", kk)
                        ddict[kk] = list()
                        dkeys.append(kk)
                else:
                    date_list.append(dtime)
                    for i,ll in enumerate(sline):
                        if i >= 1 and i <= 4:
                            ddict[dkeys[i]].append(float(ll))
                        elif i == 6:
                            ddict[dkeys[i]].append(int(ll))
                        elif i < len(dkeys):
                            ddict[dkeys[i]].append(ll)
                        else:
                            ddict[dkeys[-1]][-1] += " {:s}".format(ll)
                            
        # Create a data frame for this file
        data = pds.DataFrame(ddict, index=date_list, columns=ddict.keys())
    else:
        # Define the date parser
        def parse_smag_date(dd):                                               
            return pysat.datetime.strptime(dd, "%Y-%m-%d %H:%M:%S")

        # Load the file into a data frame
        data = pds.read_csv(fname, parse_dates={'datetime':[0]},
                            date_parser=parse_smag_date, index_col='datetime')

    return data