def HasShortCadence(EPIC, season=None):
    '''
    Returns `True` if short cadence data is available for this target.

    :param int EPIC: The EPIC ID number
    :param int season: The campaign number. Default :py:obj:`None`

    '''

    if season is None:
        season = Campaign(EPIC)
        if season is None:
            return None
    stars = GetK2Campaign(season)
    i = np.where([s[0] == EPIC for s in stars])[0]
    if len(i):
        return stars[i[0]][3]
    else:
        return None