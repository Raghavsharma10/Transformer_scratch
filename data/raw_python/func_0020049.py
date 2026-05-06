def PrimaryHDU(model):
    '''
    Construct the primary HDU file containing basic header info.

    '''

    # Get mission cards
    cards = model._mission.HDUCards(model.meta, hdu=0)
    if 'KEPMAG' not in [c[0] for c in cards]:
        cards.append(('KEPMAG', model.mag, 'Kepler magnitude'))

    # Add EVEREST info
    cards.append(('COMMENT', '************************'))
    cards.append(('COMMENT', '*     EVEREST INFO     *'))
    cards.append(('COMMENT', '************************'))
    cards.append(('MISSION', model.mission, 'Mission name'))
    cards.append(('VERSION', EVEREST_MAJOR_MINOR, 'EVEREST pipeline version'))
    cards.append(('SUBVER', EVEREST_VERSION, 'EVEREST pipeline subversion'))
    cards.append(('DATE', strftime('%Y-%m-%d'),
                  'EVEREST file creation date (YYYY-MM-DD)'))

    # Create the HDU
    header = pyfits.Header(cards=cards)
    hdu = pyfits.PrimaryHDU(header=header)

    return hdu