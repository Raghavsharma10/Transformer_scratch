def TargetDirectory(ID, season, relative=False, **kwargs):
    '''
    Returns the location of the :py:mod:`everest` data on disk
    for a given target.

    :param ID: The target ID
    :param int season: The target season number
    :param bool relative: Relative path? Default :py:obj:`False`

    '''

    if season is None:
        return None
    if relative:
        path = ''
    else:
        path = EVEREST_DAT
    return os.path.join(path, 'k2', 'c%02d' % season,
                        ('%09d' % ID)[:4] + '00000',
                        ('%09d' % ID)[4:])