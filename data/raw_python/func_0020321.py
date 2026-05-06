def Campaign(EPIC, **kwargs):
    '''
    Returns the campaign number(s) for a given EPIC target. If target
    is not found, returns :py:obj:`None`.

    :param int EPIC: The EPIC number of the target.

    '''

    campaigns = []
    for campaign, stars in GetK2Stars().items():
        if EPIC in [s[0] for s in stars]:
            campaigns.append(campaign)
    if len(campaigns) == 0:
        return None
    elif len(campaigns) == 1:
        return campaigns[0]
    else:
        return campaigns