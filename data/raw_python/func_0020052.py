def ApertureHDU(model):
    '''
    Construct the HDU containing the aperture used to de-trend.

    '''

    # Get mission cards
    cards = model._mission.HDUCards(model.meta, hdu=3)

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
    hdu = pyfits.ImageHDU(data=model.aperture,
                          header=header, name='APERTURE MASK')

    return hdu