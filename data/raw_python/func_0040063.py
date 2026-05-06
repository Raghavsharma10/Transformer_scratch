def api(feature='conditions', city='Portland', state='OR', key=None):
    """Use the wunderground API to get current conditions instead of scraping

    Please be kind and use your own key (they're FREE!):
    http://www.wunderground.com/weather/api/d/login.html

    References:
        http://www.wunderground.com/weather/api/d/terms.html

    Examples:
        >>> api('hurric', 'Boise', 'ID')  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        {u'currenthurricane': ...}}}

        >>> features = ('alerts astronomy conditions currenthurricane forecast forecast10day geolookup history hourly hourly10day ' +
        ...             'planner rawtide satellite tide webcams yesterday').split(' ')
        >> everything = [api(f, 'Portland') for f in features]
        >> js = api('alerts', 'Portland', 'OR')
        >> js = api('condit', 'Sacramento', 'CA')
        >> js = api('forecast', 'Mobile', 'AL')
        >> js = api('10day', 'Fairhope', 'AL')
        >> js = api('geo', 'Decatur', 'AL')
        >> js = api('hist', 'history', 'AL')
        >> js = api('astro')
    """
    features = ('alerts astronomy conditions currenthurricane forecast forecast10day geolookup history hourly hourly10day ' +
                'planner rawtide satellite tide webcams yesterday').split(' ')
    feature = util.fuzzy_get(features, feature)
    # Please be kind and use your own key (they're FREE!):
    # http://www.wunderground.com/weather/api/d/login.html
    key = key or env.get('WUNDERGROUND', None, verbosity=-1) or env.get('WUNDERGROUND_KEY', 'c45a86c2fc63f7d0', verbosity=-1)
    url = 'http://api.wunderground.com/api/{key}/{feature}/q/{state}/{city}.json'.format(
        key=key, feature=feature, state=state, city=city)
    return json.load(urllib.urlopen(url))