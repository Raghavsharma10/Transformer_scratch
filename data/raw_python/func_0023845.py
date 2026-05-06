def indicator_create(f, i):
    """
    Create an indicator in a feed
    :param f: feed name (eg: wes/test)
    :param i: indicator dict (eg: {'indicator': 'example.com', 'tags': ['ssh'],
    'description': 'this is a test'})
    :return: dict of indicator
    """
    if '/' not in f:
        raise ValueError('feed name must be formatted like: '
                         'csirtgadgets/scanners')

    if not i:
        raise ValueError('missing indicator dict')

    u, f = f.split('/')

    i['user'] = u
    i['feed'] = f

    ret = Indicator(i).submit()

    return ret