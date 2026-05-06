def network_from_bbox(lat_min=None, lng_min=None, lat_max=None, lng_max=None,
                      bbox=None, network_type='walk', two_way=True,
                      timeout=180, memory=None,
                      max_query_area_size=50*1000*50*1000,
                      custom_osm_filter=None):
    """
    Make a graph network from a bounding lat/lon box composed of nodes and
    edges for use in Pandana street network accessibility calculations.
    You may either enter a lat/long box via the four lat_min,
    lng_min, lat_max, lng_max parameters or the bbox parameter as a tuple.

    Parameters
    ----------
    lat_min : float
        southern latitude of bounding box, if this parameter is used the bbox
        parameter should be None.
    lng_min : float
        eastern latitude of bounding box, if this parameter is used the bbox
        parameter should be None.
    lat_max : float
        northern longitude of bounding box, if this parameter is used the bbox
        parameter should be None.
    lng_max : float
        western longitude of bounding box, if this parameter is used the bbox
        parameter should be None.
    bbox : tuple
        Bounding box formatted as a 4 element tuple:
        (lng_max, lat_min, lng_min, lat_max)
        example: (-122.304611,37.798933,-122.263412,37.822802)
        a bbox can be extracted for an area using: the CSV format bbox from
        http://boundingbox.klokantech.com/. If this parameter is used the
        lat_min, lng_min, lat_max, lng_max parameters in this function
        should be None.
    network_type : {'walk', 'drive'}, optional
        Specify the network type where value of 'walk' includes roadways where
        pedestrians are allowed and pedestrian pathways and 'drive' includes
        driveable roadways. To use a custom definition see the
        custom_osm_filter parameter. Default is walk.
    two_way : bool, optional
        Whether the routes are two-way. If True, node pairs will only
        occur once.
    timeout : int, optional
        the timeout interval for requests and to pass to Overpass API
    memory : int, optional
        server memory allocation size for the query, in bytes. If none,
        server will use its default allocation size
    max_query_area_size : float, optional
        max area for any part of the geometry, in the units the geometry is
        in: any polygon bigger will get divided up for multiple queries to
        Overpass API (default is 50,000 * 50,000 units (ie, 50km x 50km in
        area, if units are meters))
    custom_osm_filter : string, optional
        specify custom arguments for the way["highway"] query to OSM. Must
        follow Overpass API schema. For
        example to request highway ways that are service roads use:
        '["highway"="service"]'

    Returns
    -------
    nodesfinal, edgesfinal : pandas.DataFrame

    """

    start_time = time.time()

    if bbox is not None:
        assert isinstance(bbox, tuple) \
               and len(bbox) == 4, 'bbox must be a 4 element tuple'
        assert (lat_min is None) and (lng_min is None) and \
               (lat_max is None) and (lng_max is None), \
            'lat_min, lng_min, lat_max and lng_max must be None ' \
            'if you are using bbox'

        lng_max, lat_min, lng_min, lat_max = bbox

    assert lat_min is not None, 'lat_min cannot be None'
    assert lng_min is not None, 'lng_min cannot be None'
    assert lat_max is not None, 'lat_max cannot be None'
    assert lng_max is not None, 'lng_max cannot be None'
    assert isinstance(lat_min, float) and isinstance(lng_min, float) and \
        isinstance(lat_max, float) and isinstance(lng_max, float), \
        'lat_min, lng_min, lat_max, and lng_max must be floats'

    nodes, ways, waynodes = ways_in_bbox(
        lat_min=lat_min, lng_min=lng_min, lat_max=lat_max, lng_max=lng_max,
        network_type=network_type, timeout=timeout,
        memory=memory, max_query_area_size=max_query_area_size,
        custom_osm_filter=custom_osm_filter)
    log('Returning OSM data with {:,} nodes and {:,} ways...'
        .format(len(nodes), len(ways)))

    edgesfinal = node_pairs(nodes, ways, waynodes, two_way=two_way)

    # make the unique set of nodes that ended up in pairs
    node_ids = sorted(set(edgesfinal['from_id'].unique())
                      .union(set(edgesfinal['to_id'].unique())))
    nodesfinal = nodes.loc[node_ids]
    nodesfinal = nodesfinal[['lon', 'lat']]
    nodesfinal.rename(columns={'lon': 'x', 'lat': 'y'}, inplace=True)
    nodesfinal['id'] = nodesfinal.index
    edgesfinal.rename(columns={'from_id': 'from', 'to_id': 'to'}, inplace=True)
    log('Returning processed graph with {:,} nodes and {:,} edges...'
        .format(len(nodesfinal), len(edgesfinal)))
    log('Completed OSM data download and Pandana node and edge table '
        'creation in {:,.2f} seconds'.format(time.time()-start_time))

    return nodesfinal, edgesfinal