def checkNGOODPIX(filelist):
    """
    Only for ACS, WFC3 and STIS, check NGOODPIX
    If all pixels are 'bad' on all chips, exclude this image
    from further processing.
    Similar checks requiring comparing 'driz_sep_bits' against
    WFPC2 c1f.fits arrays and NICMOS DQ arrays will need to be
    done separately (and later).
    """
    toclose = False
    removed_files = []
    supported_instruments = ['ACS','STIS','WFC3']
    for inputfile in filelist:
        if isinstance(inputfile, str):
            if fileutil.getKeyword(inputfile,'instrume') in supported_instruments:
                inputfile = fits.open(inputfile)
                toclose = True
        elif inputfile[0].header['instrume'] not in supported_instruments:
            continue

        ngood = 0
        for extn in inputfile:
            if 'EXTNAME' in extn.header and extn.header['EXTNAME'] == 'SCI':
                ngood += extn.header['NGOODPIX']
        if (ngood == 0):
            removed_files.append(inputfile)
        if toclose:
            inputfile.close()

    if removed_files != []:
        print("Warning:  Files without valid pixels detected: keyword NGOODPIX = 0.0")
        print("Warning:  Removing the following files from input list")
        for f in removed_files:
            print('\t',f.filename() or "")

    return removed_files