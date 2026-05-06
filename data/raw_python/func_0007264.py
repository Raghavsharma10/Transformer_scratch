def ways_in_bbox(lat_min, lng_min, lat_max, lng_max, network_type,
                 timeout=180, memory=None,
                 max_query_area_size=50*1000*50*1000,
                 custom_osm_filter=None):
    """
    Get DataFrames of OSM data in a bounding box.

    Parameters
    ----------
    lat_min : float
        southern latitude of bounding box
    lng_min : float
        eastern longitude of bounding box
    lat_max : float
        northern latitude of bounding box
    lng_max : float
        western longitude of bounding box
    network_type : {'walk', 'drive'}, optional
        Specify the network type where value of 'walk' includes roadways
        where pedestrians are allowed and pedestrian pathways and 'drive'
        includes driveable roadways.
    timeout : int
        the timeout interval for requests and to pass to Overpass API
    memory : int
        server memory allocation size for the query, in bytes. If none,
        server will use its default allocation size
    max_query_area_size : float
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
    nodes, ways, waynodes : pandas.DataFrame

    """
    return parse_network_osm_query(
        osm_net_download(lat_max=lat_max, lat_min=lat_min, lng_min=lng_min,
                         lng_max=lng_max, network_type=network_type,
                         timeout=timeout, memory=memory,
                         max_query_area_size=max_query_area_size,
                         custom_osm_filter=custom_osm_filter))