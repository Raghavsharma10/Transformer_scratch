def HDUCards(headers, hdu=0):
    '''
    Generates HDU cards for inclusion in the de-trended light curve FITS file.
    Used internally.

    '''

    if headers is None:
        return []

    if hdu == 0:
        # Get info from the TPF Primary HDU Header
        tpf_header = headers[0]
        entries = ['TELESCOP', 'INSTRUME', 'OBJECT', 'KEPLERID', 'CHANNEL',
                   'MODULE', 'OUTPUT', 'CAMPAIGN', 'DATA_REL', 'OBSMODE',
                   'TTABLEID', 'RADESYS', 'RA_OBJ', 'DEC_OBJ',  'EQUINOX',
                   'KEPMAG']
    elif (hdu == 1) or (hdu == 6):
        # Get info from the TPF BinTable HDU Header
        tpf_header = headers[1]
        entries = ['WCSN4P', 'WCAX4P', '1CTY4P', '2CTY4P',
                   '1CUN4P', '2CUN4P', '1CRV4P',
                   '2CRV4P', '1CDL4P', '2CDL4P', '1CRP4P',
                   '2CRP4P', 'WCAX4', '1CTYP4',
                   '2CTYP4', '1CRPX4', '2CRPX4', '1CRVL4',
                   '2CRVL4', '1CUNI4', '2CUNI4',
                   '1CDLT4', '2CDLT4', '11PC4', '12PC4',
                   '21PC4', '22PC4', 'WCSN5P',
                   'WCAX5P', '1CTY5P', '2CTY5P', '1CUN5P',
                   '2CUN5P', '1CRV5P', '2CRV5P',
                   '1CDL5P', '2CDL5P', '1CRP5P', '2CRP5P',
                   'WCAX5', '1CTYP5', '2CTYP5',
                   '1CRPX5', '2CRPX5', '1CRVL5', '2CRVL5',
                   '1CUNI5', '2CUNI5', '1CDLT5',
                   '2CDLT5', '11PC5', '12PC5', '21PC5',
                   '22PC5', 'WCSN6P', 'WCAX6P',
                   '1CTY6P', '2CTY6P', '1CUN6P', '2CUN6P',
                   '1CRV6P', '2CRV6P', '1CDL6P',
                   '2CDL6P', '1CRP6P', '2CRP6P', 'WCAX6',
                   '1CTYP6', '2CTYP6', '1CRPX6',
                   '2CRPX6', '1CRVL6', '2CRVL6', '1CUNI6',
                   '2CUNI6', '1CDLT6', '2CDLT6',
                   '11PC6', '12PC6', '21PC6', '22PC6',
                   'WCSN7P', 'WCAX7P', '1CTY7P',
                   '2CTY7P', '1CUN7P', '2CUN7P', '1CRV7P',
                   '2CRV7P', '1CDL7P', '2CDL7P',
                   '1CRP7P', '2CRP7P', 'WCAX7', '1CTYP7',
                   '2CTYP7', '1CRPX7', '2CRPX7',
                   '1CRVL7', '2CRVL7', '1CUNI7', '2CUNI7',
                   '1CDLT7', '2CDLT7', '11PC7',
                   '12PC7', '21PC7', '22PC7', 'WCSN8P',
                   'WCAX8P', '1CTY8P', '2CTY8P',
                   '1CUN8P', '2CUN8P', '1CRV8P', '2CRV8P',
                   '1CDL8P', '2CDL8P', '1CRP8P',
                   '2CRP8P', 'WCAX8', '1CTYP8', '2CTYP8',
                   '1CRPX8', '2CRPX8', '1CRVL8',
                   '2CRVL8', '1CUNI8', '2CUNI8', '1CDLT8',
                   '2CDLT8', '11PC8', '12PC8',
                   '21PC8', '22PC8', 'WCSN9P', 'WCAX9P',
                   '1CTY9P', '2CTY9P', '1CUN9P',
                   '2CUN9P', '1CRV9P', '2CRV9P', '1CDL9P',
                   '2CDL9P', '1CRP9P', '2CRP9P',
                   'WCAX9', '1CTYP9', '2CTYP9', '1CRPX9',
                   '2CRPX9', '1CRVL9', '2CRVL9',
                   '1CUNI9', '2CUNI9', '1CDLT9', '2CDLT9',
                   '11PC9', '12PC9', '21PC9',
                   '22PC9', 'INHERIT', 'EXTNAME', 'EXTVER',
                   'TELESCOP', 'INSTRUME',
                   'OBJECT', 'KEPLERID', 'RADESYS', 'RA_OBJ',
                   'DEC_OBJ', 'EQUINOX',
                   'EXPOSURE', 'TIMEREF', 'TASSIGN', 'TIMESYS',
                   'BJDREFI', 'BJDREFF',
                   'TIMEUNIT', 'TELAPSE', 'LIVETIME', 'TSTART',
                   'TSTOP', 'LC_START',
                   'LC_END', 'DEADC', 'TIMEPIXR', 'TIERRELA',
                   'INT_TIME', 'READTIME',
                   'FRAMETIM', 'NUM_FRM', 'TIMEDEL', 'DATE-OBS',
                   'DATE-END', 'BACKAPP',
                   'DEADAPP', 'VIGNAPP', 'GAIN', 'READNOIS',
                   'NREADOUT', 'TIMSLICE',
                   'MEANBLCK', 'LCFXDOFF', 'SCFXDOFF']
    elif (hdu == 3) or (hdu == 4) or (hdu == 5):
        # Get info from the TPF BinTable HDU Header
        tpf_header = headers[2]
        entries = ['TELESCOP', 'INSTRUME', 'OBJECT', 'KEPLERID', 'RADESYS',
                   'RA_OBJ', 'DEC_OBJ', 'EQUINOX', 'WCSAXES', 'CTYPE1',
                   'CTYPE2', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2',
                   'CUNIT1', 'CUNIT2', 'CDELT1', 'CDELT2', 'PC1_1',
                   'PC1_2', 'PC2_1', 'PC2_2', 'WCSNAMEP', 'WCSAXESP',
                   'CTYPE1P', 'CUNIT1P', 'CRPIX1P', 'CRVAL1P', 'CDELT1P',
                   'CTYPE2P', 'CUNIT2P', 'CRPIX2P', 'CRVAL2P', 'CDELT2P',
                   'NPIXSAP', 'NPIXMISS']
    else:
        return []

    cards = []
    cards.append(('COMMENT', '************************'))
    cards.append(('COMMENT', '*     MISSION INFO     *'))
    cards.append(('COMMENT', '************************'))
    for entry in entries:
        try:
            cards.append(tuple(tpf_header[entry]))
        except KeyError:
            pass
    return cards