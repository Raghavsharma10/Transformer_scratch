def load_ini(filename):
    """
    Read a CLASS ``.ini`` file, returning a dictionary of parameters

    Parameters
    ----------
    filename : str
        the name of an existing parameter file to load, or one included as
        part of the CLASS source

    Returns
    -------
    dict :
        the input parameters loaded from file
    """
    # also look in data dir
    path = _find_file(filename)

    pars = {}
    with open(path, 'r') as ff:

        # loop over lines
        for lineno, line in enumerate(ff):
            if not line: continue

            # skip any commented lines with #
            if '#' in line: line = line[line.index('#')+1:]

            # must have an equals sign to be valid
            if "=" not in line: continue

            # extract key and value pairs
            fields = line.split("=")
            if len(fields) != 2:
                import warnings
                warnings.warn("skipping line number %d: '%s'" %(lineno,line))
                continue
            pars[fields[0].strip()] = fields[1].strip()

    return pars