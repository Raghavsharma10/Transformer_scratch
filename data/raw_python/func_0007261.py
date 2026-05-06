def process_node(e):
    """
    Process a node element entry into a dict suitable for going into a
    Pandas DataFrame.

    Parameters
    ----------
    e : dict
        individual node element in downloaded OSM json

    Returns
    -------
    node : dict

    """
    node = {'id': e['id'],
            'lat': e['lat'],
            'lon': e['lon']}

    if 'tags' in e:
        if e['tags'] is not np.nan:
            for t, v in list(e['tags'].items()):
                if t in config.settings.keep_osm_tags:
                    node[t] = v

    return node