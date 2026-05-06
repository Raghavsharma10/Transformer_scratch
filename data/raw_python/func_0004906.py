def readspec(filename, read_scan=None):
    """Open a SPEC file and read its content

    Inputs:

        filename: string
            the file to open

        read_scan: None, 'all' or integer
            the index of scan to be read from the file. If None, no scan should be read. If
            'all', all scans should be read. If a number, just the scan with that number
            should be read.

    Output:
        the data in the spec file in a dict.
    """
    with open(filename, 'rt') as f:
        sf = {'motors': [], 'maxscannumber': 0}
        sf['originalfilename'] = filename
        lastscannumber = None
        while True:
            l = f.readline()
            if l.startswith('#F'):
                sf['filename'] = l[2:].strip()
            elif l.startswith('#E'):
                sf['epoch'] = int(l[2:].strip())
                sf['datetime'] = datetime.datetime.fromtimestamp(sf['epoch'])
            elif l.startswith('#D'):
                sf['datestring'] = l[2:].strip()
            elif l.startswith('#C'):
                sf['comment'] = l[2:].strip()
            elif l.startswith('#O'):
                try:
                    l = l.split(None, 1)[1]
                except IndexError:
                    continue
                if 'motors' not in list(sf.keys()):
                    sf['motors'] = []
                sf['motors'].extend([x.strip() for x in l.split('  ')])
            elif not l.strip():
                # empty line, signifies the end of the header part. The next
                # line will be a scan.
                break
        sf['scans'] = {}
        if read_scan is not None:
            if read_scan == 'all':
                nr = None
            else:
                nr = read_scan
            try:
                while True:
                    s = readspecscan(f, nr)
                    if isinstance(s, dict):
                        sf['scans'][s['number']] = s
                        if nr is not None:
                            break
                        sf['maxscannumber'] = max(
                            sf['maxscannumber'], s['number'])
                    elif s is not None:
                        sf['maxscannumber'] = max(sf['maxscannumber'], s)
            except SpecFileEOF:
                pass
        else:
            while True:
                l = f.readline()
                if not l:
                    break
                if l.startswith('#S'):
                    n = int(l[2:].split()[0])
                    sf['maxscannumber'] = max(sf['maxscannumber'], n)
        for n in sf['scans']:
            s = sf['scans'][n]
            s['motors'] = sf['motors']
            if 'comment' not in s:
                s['comment'] = sf['comment']
            if 'positions' not in s:
                s['positions'] = [None] * len(sf['motors'])
    return sf