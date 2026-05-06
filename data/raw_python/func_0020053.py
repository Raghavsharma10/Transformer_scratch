def ImagesHDU(model):
    '''
    Construct the HDU containing sample postage stamp images of the target.

    '''

    # Get mission cards
    cards = model._mission.HDUCards(model.meta, hdu=4)

    # Add EVEREST info
    cards.append(('COMMENT', '************************'))
    cards.append(('COMMENT', '*     EVEREST INFO     *'))
    cards.append(('COMMENT', '************************'))
    cards.append(('MISSION', model.mission, 'Mission name'))
    cards.append(('VERSION', EVEREST_MAJOR_MINOR, 'EVEREST pipeline version'))
    cards.append(('SUBVER', EVEREST_VERSION, 'EVEREST pipeline subversion'))
    cards.append(('DATE', strftime('%Y-%m-%d'),
                  'EVEREST file creation date (YYYY-MM-DD)'))

    # The images
    format = '%dD' % model.pixel_images[0].shape[1]
    arrays = [pyfits.Column(name='STAMP1', format=format,
                            array=model.pixel_images[0]),
              pyfits.Column(name='STAMP2', format=format,
                            array=model.pixel_images[1]),
              pyfits.Column(name='STAMP3', format=format,
                            array=model.pixel_images[2])]

    # Create the HDU
    header = pyfits.Header(cards=cards)
    cols = pyfits.ColDefs(arrays)
    hdu = pyfits.BinTableHDU.from_columns(
        cols, header=header, name='POSTAGE STAMPS')

    return hdu