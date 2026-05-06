def readbhfv1(filename, load_data=False, bdfext='.bdf', bhfext='.bhf'):
    """Read header data from bdf/bhf file (Bessy Data Format v1)

    Input:
        filename: the name of the file
        load_data: if the matrices are to be loaded

    Output:
        bdf: the BDF header structure

    Adapted the bdf_read.m macro from Sylvio Haas.
    """
    # strip the bhf or bdf extension if there.
    if filename.endswith(bdfext):
        basename = filename[:-len(bdfext)]
    elif filename.endswith(bhfext):
        basename = filename[:-len(bhfext)]
    else:  # assume a single file of header and data.
        basename, bhfext = os.path.splitext(filename)
        bdfext = bhfext
    headername = basename + bhfext
    dataname = basename + bdfext
    bdf = {}
    bdf['his'] = []  # empty list for history
    bdf['C'] = {}  # empty list for bdf file descriptions
    namelists = {}
    valuelists = {}
    with open(headername, 'rb') as fid:  # if fails, an exception is raised
        for line in fid:
            if not line.strip():
                continue  # empty line
            mat = line.split(None, 1)
            prefix = mat[0]
            if prefix == '#C':
                left, right = mat[1].split('=', 1)
                left = left.strip()
                right = right.strip()
                if left in ['xdim', 'ydim']:
                    bdf[left] = int(right)
                elif left in ['type', 'bdf']:
                    bdf[left] = right
                if left in ['Sendtime']:
                    bdf['C'][left] = float(right)
                elif left in ['xdim', 'ydim']:
                    bdf['C'][left] = int(right)
                else:
                    bdf['C'][left] = misc.parse_number(right)
            elif prefix.startswith("#H"):
                bdf['his'].append(mat[1])
#            elif prefix.startswith("#DATA"):
#                if not load_data:
#                    break
#                darray = np.fromfile(fid, dtype = bdf['type'], count = int(bdf['xdim'] * bdf['ydim']))
#                bdf['data'] = np.rot90((darray.reshape(bdf['xdim'], bdf['ydim'])).astype('double').T, 1).copy() # this weird transformation is needed to get the matrix in the same form as bdf_read.m gets it.
#            elif prefix.startswith('#ERROR'):
#                if not load_data:
#                    break
#                darray = np.fromfile(fid, dtype = bdf['type'], count = int(bdf['xdim'] * bdf['ydim']))
#                bdf['error'] = np.rot90((darray.reshape(bdf['xdim'], bdf['ydim'])).astype('double').T, 1).copy()
            else:
                for prf in ['M', 'G', 'S', 'T']:
                    if prefix.startswith('#C%sL' % prf):
                        if prf not in namelists:
                            namelists[prf] = []
                        namelists[prf].extend(mat[1].split())
                    elif prefix.startswith('#C%sV' % prf):
                        if prf not in valuelists:
                            valuelists[prf] = []
                        valuelists[prf].extend([float(x)
                                                for x in mat[1].split()])
                    else:
                        continue
    for dictname, prfname in zip(['M', 'CG', 'CS', 'CT'], ['M', 'G', 'S', 'T']):
        bdf[dictname] = dict(
            list(zip(namelists[prfname], valuelists[prfname])))
    bdf['__Origin__'] = 'BDFv1'
    bdf['__particle__'] = 'photon'

    if load_data:
        f = open(dataname, 'r')
        try:
            s = f.read()
        except IOError as ioe:
            # an ugly bug (M$ KB899149) in W!nd0w$ causes an error if loading too
            # large a file from a network drive and opening it read-only.
            if ioe.errno == 22:
                f.close()
                try:
                    # one work-around is to open it read-write.
                    f = open(dataname, 'r+b')
                    s = f.read()
                except IOError:
                    # if this does not work, inform the user to either obtain
                    # write permission for that file or copy it to a local
                    # drive
                    f.close()
                    raise IOError(22, """
You were probably trying to open a read-only file from a network drive on
Windows, weren\'t you? There is a bug in Windows causing this error
(see http://support.microsoft.com/default.aspx?scid=kb;en-us;899149).
To work around this, please either obtain write permission for that file
(I won't write anything to it, I promise!!!) or copy it to a local drive.
Sorry for the inconvenience.""", ioe.filename)
        datasets = re.findall(
            '#\s*(?P<name>\w+)\[(?P<xsize>\d+):(?P<ysize>\d+)\]', s)
        names = [d[0] for d in datasets]
        xsize = [int(d[1]) for d in datasets]
        ysize = [int(d[2]) for d in datasets]
        dt = np.dtype(bdf['type'])
        for i in range(len(datasets)):
            start = s.find('#%s' % names[i])
            if i < len(datasets) - 1:
                end = s.find('#%s' % (names[i + 1]))
            else:
                end = len(s)
            s1 = s[start:end]
            datasize = xsize[i] * ysize[i] * dt.itemsize
            if datasize > len(s1):
                # assume we are dealing with a BOOL matrix
                bdf[names[i]] = np.fromstring(
                    s1[-xsize[i] * ysize[i]:], dtype=np.uint8)
            else:
                bdf[names[i]] = np.fromstring(
                    s1[-xsize[i] * ysize[i] * dt.itemsize:], dtype=dt)
            # conversion: Matlab saves the array in Fortran-style ordering (columns first).
            # Python however loads in C-style: rows first. We need to take care:
            #   1) reshape from linear to (ysize,xsize) and not (xsize,ysize)
            #   2) transpose (swaps columns and rows)
            # After these operations, we only have to rotate this counter-clockwise by 90
            # degrees because bdf2_write rotates by +270 degrees before saving.
            bdf[names[i]] = np.rot90(
                bdf[names[i]].reshape((ysize[i], xsize[i]), order='F'), 1)

    return bdf