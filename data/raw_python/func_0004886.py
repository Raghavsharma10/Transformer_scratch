def readehf(filename):
    """Read EDF header (ESRF data format, as of beamline ID01 and ID02)

    Input
    -----
    filename: string
        the file name to load

    Output
    ------
    the EDF header structure in a dictionary
    """
    f = open(filename, 'r')
    edf = {}
    if not f.readline().strip().startswith('{'):
        raise ValueError('Invalid file format.')
    for l in f:
        l = l.strip()
        if not l:
            continue
        if l.endswith('}'):
            break  # last line of header
        try:
            left, right = l.split('=', 1)
        except ValueError:
            raise ValueError('Invalid line: ' + l)
        left = left.strip()
        right = right.strip()
        if not right.endswith(';'):
            raise ValueError(
                'Invalid line (does not end with a semicolon): ' + l)
        right = right[:-1].strip()
        m = re.match('^(?P<left>.*)~(?P<continuation>\d+)$', left)
        if m is not None:
            edf[m.group('left')] = edf[m.group('left')] + right
        else:
            edf[left] = _readedf_extractline(left, right)
    f.close()
    edf['FileName'] = filename
    edf['__Origin__'] = 'EDF ID02'
    edf['__particle__'] = 'photon'
    return edf