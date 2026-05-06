def PixelsHDU(model):
    '''
    Construct the HDU containing the pixel-level light curve.

    '''

    # Get mission cards
    cards = model._mission.HDUCards(model.meta, hdu=2)

    # Add EVEREST info
    cards = []
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

    # The pixel timeseries
    arrays = [pyfits.Column(name='FPIX', format='%dD' %
                            model.fpix.shape[1], array=model.fpix)]

    # The first order PLD vectors for all the neighbors (npixels, ncadences)
    X1N = model.X1N
    if X1N is not None:
        arrays.append(pyfits.Column(name='X1N', format='%dD' %
                                    X1N.shape[1], array=X1N))

    cols = pyfits.ColDefs(arrays)
    hdu = pyfits.BinTableHDU.from_columns(cols, header=header, name='PIXELS')

    return hdu