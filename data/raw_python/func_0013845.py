def append_ascii_data(file_strings, tag):
    """ Append data from multiple files for the same time period

    Parameters
    -----------
    file_strings : array-like
        Lists or arrays of strings, where each string contains one file of data
    tag : string
        String denoting the type of file to load, accepted values are 'indices',
        'all', 'stations', and None (for only magnetometer data)

    Returns
    -------
    out_string : string
        String with all data, ready for output to a file
        
    """
    import re
    
    # Start with data from the first list element
    out_lines = file_strings[0].split('\n')
    iparam = -1 # Index for the parameter line
    ihead = -1 # Index for the last header line
    idates = list() # Indices for the date lines
    date_list = list() # List of dates
    num_stations = list() # Number of stations for each date line
    ind_num = 2 if tag in ['all', 'indices', ''] else 0
    # ind_num = 2 if tag == '' else ind_num

    # Find the index information for the data
    for i,line in enumerate(out_lines):
        if line == "Selected parameters:":
            iparam = i + 1
        elif line.count("=") == len(line) and len(line) > 2:
            ihead = i
            break

    # Find the time indices and number of stations for each date line
    i = ihead + 1
    while i < len(out_lines) - 1:
        idates.append(i)
        lsplit = re.split('\t+', out_lines[i])
        dtime = pds.datetime.strptime(" ".join(lsplit[0:-1]),
                                      "%Y %m %d %H %M %S")
        date_list.append(dtime)
        num_stations.append(int(lsplit[-1]))
        i += num_stations[-1] + 1 + ind_num
    idates = np.array(idates)

    # Initialize a list of station names
    station_names = list()
    
    # Cycle through each additional set of file strings
    for ff in range(len(file_strings)-1):
        file_lines = file_strings[ff+1].split('\n')

        # Find the index information for the data
        head = True
        snum = 0
        for i,line in enumerate(file_lines):
            if head:
                if line.count("=") == len(line) and len(line) > 2:
                    head = False
            elif len(line) > 0:
                lsplit = re.split('\t+', line)
                if snum == 0:
                    dtime = pds.datetime.strptime(" ".join(lsplit[0:-1]),
                                                  "%Y %m %d %H %M %S")
                    try:
                        idate = date_list.index(dtime)
                    except:
                        # SuperMAG outputs date lines regardless of the
                        # number of stations.  These files shouldn't be
                        # appended together.
                        raise ValueError("Unexpected date ", dtime)

                    snum = int(lsplit[-1])
                    onum = num_stations[idate]
                    inum = ind_num

                    # Adjust reference data for new number of station lines
                    idates[idate+1:] += snum
                    num_stations[idate] += snum

                    # Adjust date line for new number of station lines
                    oline = "{:s}\t{:d}".format( \
                                    dtime.strftime("%Y\t%m\t%d\t%H\t%M\t%S"),
                                                 num_stations[idate])
                    out_lines[idates[idate]] = oline
                else:
                    if inum > 0:
                        inum -= 1
                    else:
                        # Insert the station line to the end of the date section
                        onum += 1
                        snum -= 1
                        out_lines.insert(idates[idate]+onum, line)

                        # Save the station name to update the parameter line
                        if not lsplit[0] in station_names:
                            station_names.append(lsplit[0])

    # Update the parameter line
    out_lines[iparam] += "," + ",".join(station_names)

    # Join the output lines into a single string
    out_string = "\n".join(out_lines)

    return out_string