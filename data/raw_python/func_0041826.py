def fmtval(value, colorstr=None, precision=None, spacing=True, trunc=True,
           end=' '):
    ''' Formats and returns a given number according to specifications. '''
    colwidth = opts.colwidth
    # get precision
    if precision is None:
        precision = opts.precision
    fmt = '%%.%sf' % precision

    # format with decimal mark, separators
    result = locale.format(fmt, value, True)

    if spacing:
        result = '%%%ss' % colwidth % result

    if trunc:
        if len(result) > colwidth:   # truncate w/ellipsis
            result = truncstr(result, colwidth)

    # Add color if needed
    if opts.incolor and colorstr:
        return colorstr % result + end
    else:
        return result + end