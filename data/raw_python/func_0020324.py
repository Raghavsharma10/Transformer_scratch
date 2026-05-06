def Channel(EPIC, campaign=None):
    '''
    Returns the channel number for a given EPIC target.

    '''

    if campaign is None:
        campaign = Campaign(EPIC)
    if hasattr(campaign, '__len__'):
        raise AttributeError(
            "Please choose a campaign/season for this target: %s." % campaign)
    try:
        stars = GetK2Stars()[campaign]
    except KeyError:
        # Not sure what else to do here!
        log.warn("Unknown channel for target. Defaulting to channel 2.")
        return 2
    i = np.argmax([s[0] == EPIC for s in stars])
    return stars[i][2]