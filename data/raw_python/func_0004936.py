def writebdfv2(filename, bdf, bdfext='.bdf', bhfext='.bhf'):
    """Write a version 2 Bessy Data File

    Inputs
    ------
    filename: string
        the name of the output file. One can give the complete header or
        datafile name or just the base name without the extensions.
    bdf: dict
        the BDF structure (in the same format as loaded by ``readbdfv2()``
    bdfext: string, optional
        the extension of the data file
    bhfext: string, optional
        the extension of the header file

    Output
    ------
    None

    Notes
    -----
    BDFv2 header and scattering data are stored separately in the header and
    the data files. Given the file name both are saved.
    """
    if filename.endswith(bdfext):
        basename = filename[:-len(bdfext)]
    elif filename.endswith(bhfext):
        basename = filename[:-len(bhfext)]
    else:
        basename = filename
    header.writebhfv2(basename + '.bhf', bdf)
    f = open(basename + '.bdf', 'wb')
    keys = ['RAWDATA', 'RAWERROR', 'CORRDATA', 'CORRERROR', 'NANDATA']
    keys.extend(
        [x for x in list(bdf.keys()) if isinstance(bdf[x], np.ndarray) and x not in keys])
    for k in keys:
        if k not in list(bdf.keys()):
            continue
        f.write('#%s[%d:%d]\n' % (k, bdf['xdim'], bdf['ydim']))
        f.write(np.rot90(bdf[k], 3).astype('float32').tostring(order='F'))
    f.close()