def SaturationFlux(EPIC, campaign=None, **kwargs):
    '''
    Returns the well depth for the target. If any of the target's pixels
    have flux larger than this value, they are likely to be saturated and
    cause charge bleeding. The well depths were obtained from Table 13
    of the Kepler instrument handbook. We assume an exposure time of 6.02s.

    '''

    channel, well_depth = np.loadtxt(os.path.join(EVEREST_SRC, 'missions',
                                                  'k2',
                                                  'tables', 'well_depth.tsv'),
                                     unpack=True)
    satflx = well_depth[channel == Channel(EPIC, campaign=campaign)][0] / 6.02
    return satflx