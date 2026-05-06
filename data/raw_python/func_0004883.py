def writeB1logfile(filename, data):
    """Write a header structure into a B1 logfile.

    Inputs:
        filename: name of the file.
        data: header dictionary

    Notes:
        exceptions pass through to the caller.
    """
    allkeys = list(data.keys())
    f = open(filename, 'wt', encoding='utf-8')
    for ld in _logfile_data:  # process each line
        linebegin = ld[0]
        fieldnames = ld[1]
        # set the default formatter if it is not given
        if len(ld) < 3:
            formatter = str
        elif ld[2] is None:
            formatter = str
        else:
            formatter = ld[2]
        # this will contain the formatted values.
        formatted = ''
        if isinstance(fieldnames, str):
            # scalar field name, just one field. Formatter should be a
            # callable.
            if fieldnames not in allkeys:
                # this field has already been processed
                continue
            try:
                formatted = formatter(data[fieldnames])
            except KeyError:
                # field not found in param structure
                continue
        elif isinstance(fieldnames, tuple):
            # more than one field names in a tuple. In this case, formatter can
            # be a tuple of callables...
            if all([(fn not in allkeys) for fn in fieldnames]):
                # if all the fields have been processed:
                continue
            if isinstance(formatter, tuple) and len(formatter) == len(fieldnames):
                formatted = ' '.join([ft(data[fn])
                                      for ft, fn in zip(formatter, fieldnames)])
            # ...or a single callable...
            elif not isinstance(formatter, tuple):
                formatted = formatter([data[fn] for fn in fieldnames])
            # ...otherwise raise an exception.
            else:
                raise SyntaxError('Programming error: formatter should be a scalar or a tuple\
of the same length as the field names in logfile_data.')
        else:  # fieldnames is neither a string, nor a tuple.
            raise SyntaxError(
                'Invalid syntax (programming error) in logfile_data in writeparamfile().')
        # try to get the values
        linetowrite = linebegin + ':\t' + formatted + '\n'
        f.write(linetowrite)
        if isinstance(fieldnames, tuple):
            for fn in fieldnames:  # remove the params treated.
                if fn in allkeys:
                    allkeys.remove(fn)
        else:
            if fieldnames in allkeys:
                allkeys.remove(fieldnames)
    # write untreated params
    for k in allkeys:
        linetowrite = k + ':\t' + str(data[k]) + '\n'
        f.write(linetowrite)

    f.close()