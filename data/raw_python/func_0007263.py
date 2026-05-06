def parse_network_osm_query(data):
    """
    Convert OSM query data to DataFrames of ways and way-nodes.

    Parameters
    ----------
    data : dict
        Result of an OSM query.

    Returns
    -------
    nodes, ways, waynodes : pandas.DataFrame

    """
    if len(data['elements']) == 0:
        raise RuntimeError('OSM query results contain no data.')

    nodes = []
    ways = []
    waynodes = []

    for e in data['elements']:
        if e['type'] == 'node':
            nodes.append(process_node(e))
        elif e['type'] == 'way':
            w, wn = process_way(e)
            ways.append(w)
            waynodes.extend(wn)

    nodes = pd.DataFrame.from_records(nodes, index='id')
    ways = pd.DataFrame.from_records(ways, index='id')
    waynodes = pd.DataFrame.from_records(waynodes, index='way_id')

    return (nodes, ways, waynodes)