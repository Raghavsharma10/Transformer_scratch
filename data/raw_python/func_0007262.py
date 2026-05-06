def process_way(e):
    """
    Process a way element entry into a list of dicts suitable for going into
    a Pandas DataFrame.

    Parameters
    ----------
    e : dict
        individual way element in downloaded OSM json

    Returns
    -------
    way : dict
    waynodes : list of dict

    """
    way = {'id': e['id']}

    if 'tags' in e:
        if e['tags'] is not np.nan:
            for t, v in list(e['tags'].items()):
                if t in config.settings.keep_osm_tags:
                    way[t] = v

    # nodes that make up a way
    waynodes = []

    for n in e['nodes']:
        waynodes.append({'way_id': e['id'], 'node_id': n})

    return way, waynodes