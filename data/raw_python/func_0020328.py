def RemoveBackground(EPIC, campaign=None):
    '''
    Returns :py:obj:`True` or :py:obj:`False`, indicating whether or not
    to remove the background flux for the target. If ``campaign < 3``,
    returns :py:obj:`True`, otherwise returns :py:obj:`False`.

    '''

    if campaign is None:
        campaign = Campaign(EPIC)
    if hasattr(campaign, '__len__'):
        raise AttributeError(
            "Please choose a campaign/season for this target: %s." % campaign)
    if campaign < 3:
        return True
    else:
        return False