def feed(f, limit=25):
    """
    Pull a feed
    :param f: feed name (eg: csirtgadgetes/correlated)
    :param limit: return value limit (default 25)
    :return: Feed dict
    """
    if '/' not in f:
        raise ValueError('feed name must be formatted like: '
                         'csirtgadgets/scanners')

    user, f = f.split('/')

    return Feed().show(user, f, limit=limit)