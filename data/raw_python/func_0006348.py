def get_parameter_from_files(files, parameters=None, unique=False, sort=True):
    ''' Takes a list of files, searches for the parameter name in the file name and in the file.
    Returns a ordered dict with the file name in the first dimension and the corresponding parameter values in the second.
    If a scan parameter appears in the file name and in the file the first parameter setting has to be in the file name, otherwise a warning is shown.
    The file names can be sorted by the first parameter value of each file.

    Parameters
    ----------
    files : string, list of strings
    parameters : string, list of strings
    unique : boolean
        If set only one file per scan parameter value is used.
    sort : boolean

    Returns
    -------
    collections.OrderedDict

    '''
    logging.debug('Get the parameter ' + str(parameters) + ' values from ' + str(len(files)) + ' files')
    files_dict = collections.OrderedDict()
    if isinstance(files, basestring):
        files = (files, )
    if isinstance(parameters, basestring):
        parameters = (parameters, )
    parameter_values_from_file_names_dict = get_parameter_value_from_file_names(files, parameters, unique=unique, sort=sort)  # get the parameter from the file name
    for file_name in files:
        with tb.open_file(file_name, mode="r") as in_file_h5:  # open the actual file
            scan_parameter_values = collections.OrderedDict()
            try:
                scan_parameters = in_file_h5.root.scan_parameters[:]  # get the scan parameters from the scan parameter table
                if parameters is None:
                    parameters = get_scan_parameter_names(scan_parameters)
                for parameter in parameters:
                    try:
                        scan_parameter_values[parameter] = np.unique(scan_parameters[parameter]).tolist()  # different scan parameter values used
                    except ValueError:  # the scan parameter does not exists
                        pass
            except tb.NoSuchNodeError:  # scan parameter table does not exist
                try:
                    scan_parameters = get_scan_parameter(in_file_h5.root.meta_data[:])  # get the scan parameters from the meta data
                    if scan_parameters:
                        try:
                            scan_parameter_values = np.unique(scan_parameters[parameters]).tolist()  # different scan parameter values used
                        except ValueError:  # the scan parameter does not exists
                            pass
                except tb.NoSuchNodeError:  # meta data table does not exist
                    pass
            if not scan_parameter_values:  # if no scan parameter values could be set from file take the parameter found in the file name
                try:
                    scan_parameter_values = parameter_values_from_file_names_dict[file_name]
                except KeyError:  # no scan parameter found at all, neither in the file name nor in the file
                    scan_parameter_values = None
            else:  # use the parameter given in the file and cross check if it matches the file name parameter if these is given
                try:
                    for key, value in scan_parameter_values.items():
                        if value and value[0] != parameter_values_from_file_names_dict[file_name][key][0]:  # parameter value exists: check if the first value is the file name value
                            logging.warning('Parameter values in the file name and in the file differ. Take ' + str(key) + ' parameters ' + str(value) + ' found in %s.', file_name)
                except KeyError:  # parameter does not exists in the file name
                    pass
                except IndexError:
                    raise IncompleteInputError('Something wrong check!')
            if unique and scan_parameter_values is not None:
                existing = False
                for parameter in scan_parameter_values:  # loop to determine if any value of any scan parameter exists already
                    all_par_values = [values[parameter] for values in files_dict.values()]
                    if any(x in [scan_parameter_values[parameter]] for x in all_par_values):
                        existing = True
                        break
                if not existing:
                    files_dict[file_name] = scan_parameter_values
                else:
                    logging.warning('Scan parameter value(s) from %s exists already, do not add to result', file_name)
            else:
                files_dict[file_name] = scan_parameter_values
    return collections.OrderedDict(sorted(files_dict.iteritems(), key=itemgetter(1)) if sort else files_dict)