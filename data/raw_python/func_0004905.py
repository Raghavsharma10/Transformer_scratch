def readspecscan(f, number=None):
    """Read the next spec scan in the file, which starts at the current position."""
    scan = None
    scannumber = None
    while True:
        l = f.readline()
        if l.startswith('#S'):
            scannumber = int(l[2:].split()[0])
            if not ((number is None) or (number == scannumber)):
                # break the loop, will skip to the next empty line after this
                # loop
                break
            if scan is None:
                scan = {}
            scan['number'] = scannumber
            scan['command'] = l[2:].split(None, 1)[1].strip()
            scan['data'] = []
        elif l.startswith('#C'):
            scan['comment'] = l[2:].strip()
        elif l.startswith('#D'):
            scan['datestring'] = l[2:].strip()
        elif l.startswith('#T'):
            scan['countingtime'] = float(l[2:].split()[0])
            scan['scantimeunits'] = l[2:].split()[1].strip()
        elif l.startswith('#M'):
            scan['countingcounts'] = float(l[2:].split()[0])
        elif l.startswith('#G'):
            if 'G' not in scan:
                scan['G'] = []
            scan['G'].extend([float(x) for x in l.split()[1:]])
        elif l.startswith('#P'):
            if 'positions' not in scan:
                scan['positions'] = []
            scan['positions'].extend([float(x) for x in l.split()[1:]])
        elif l.startswith('#Q'):
            pass
        elif l.startswith('#N'):
            n = [float(x) for x in l[2:].strip().split()]
            if len(n) == 1:
                scan['N'] = n[0]
            else:
                scan['N'] = n
        elif l.startswith('#L'):
            scan['Columns'] = [x.strip() for x in l[3:].split('  ')]
        elif not l:
            # end of file
            if scan is None:
                raise SpecFileEOF
            else:
                break
        elif not l.strip():
            break  # empty line, end of scan in file.
        elif l.startswith('#'):
            # ignore other lines starting with a hashmark.
            continue
        else:
            scan['data'].append(tuple(float(x) for x in l.split()))
    while l.strip():
        l = f.readline()
    if scan is not None:
        scan['data'] = np.array(
            scan['data'], dtype=list(zip(scan['Columns'], itertools.repeat(np.float))))
        return scan
    else:
        return scannumber