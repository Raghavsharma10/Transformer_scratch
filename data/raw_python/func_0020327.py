def KepMag(EPIC, campaign=None):
    '''
    Returns the *Kepler* magnitude for a given EPIC target.

    '''

    if campaign is None:
        campaign = Campaign(EPIC)
    if hasattr(campaign, '__len__'):
        raise AttributeError(
            "Please choose a campaign/season for this target: %s." % campaign)
    stars = GetK2Stars()[campaign]
    i = np.argmax([s[0] == EPIC for s in stars])
    return stars[i][1]