def osm_net_download(lat_min=None, lng_min=None, lat_max=None, lng_max=None,
                     network_type='walk', timeout=180, memory=None,
                     max_query_area_size=50*1000*50*1000,
                     custom_osm_filter=None):
    """
    Download OSM ways and nodes within a bounding box from the Overpass API.

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
    network_type : string
        Specify the network type where value of 'walk' includes roadways
        where pedestrians are allowed and pedestrian
        pathways and 'drive' includes driveable roadways.
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
    response_json : dict
    """

    # create a filter to exclude certain kinds of ways based on the requested
    # network_type
    if custom_osm_filter is None:
        request_filter = osm_filter(network_type)
    else:
        request_filter = custom_osm_filter

    response_jsons_list = []
    response_jsons = []

    # server memory allocation in bytes formatted for Overpass API query
    if memory is None:
        maxsize = ''
    else:
        maxsize = '[maxsize:{}]'.format(memory)

    # define the Overpass API query
    # way["highway"] denotes ways with highway keys and {filters} returns
    # ways with the requested key/value. the '>' makes it recurse so we get
    # ways and way nodes. maxsize is in bytes.

    # turn bbox into a polygon and project to local UTM
    polygon = Polygon([(lng_max, lat_min), (lng_min, lat_min),
                       (lng_min, lat_max), (lng_max, lat_max)])
    geometry_proj, crs_proj = project_geometry(polygon,
                                               crs={'init': 'epsg:4326'})

    # subdivide the bbox area poly if it exceeds the max area size
    # (in meters), then project back to WGS84
    geometry_proj_consolidated_subdivided = consolidate_subdivide_geometry(
        geometry_proj, max_query_area_size=max_query_area_size)
    geometry, crs = project_geometry(geometry_proj_consolidated_subdivided,
                                     crs=crs_proj, to_latlong=True)
    log('Requesting network data within bounding box from Overpass API '
        'in {:,} request(s)'.format(len(geometry)))
    start_time = time.time()

    # loop through each polygon in the geometry
    for poly in geometry:
        # represent bbox as lng_max, lat_min, lng_min, lat_max and round
        # lat-longs to 8 decimal places to create
        # consistent URL strings
        lng_max, lat_min, lng_min, lat_max = poly.bounds
        query_template = '[out:json][timeout:{timeout}]{maxsize};' \
                         '(way["highway"]' \
                         '{filters}({lat_min:.8f},{lng_max:.8f},' \
                         '{lat_max:.8f},{lng_min:.8f});>;);out;'
        query_str = query_template.format(lat_max=lat_max, lat_min=lat_min,
                                          lng_min=lng_min, lng_max=lng_max,
                                          filters=request_filter,
                                          timeout=timeout, maxsize=maxsize)
        response_json = overpass_request(data={'data': query_str},
                                         timeout=timeout)

        response_jsons_list.append(response_json)

    log('Downloaded OSM network data within bounding box from Overpass '
        'API in {:,} request(s) and'
        ' {:,.2f} seconds'.format(len(geometry), time.time()-start_time))

    # stitch together individual json results
    for json in response_jsons_list:
        try:
            response_jsons.extend(json['elements'])
        except KeyError:
            pass

    # remove duplicate records resulting from the json stitching
    start_time = time.time()
    record_count = len(response_jsons)

    if record_count == 0:
        raise Exception('Query resulted in no data. Check your query '
                        'parameters: {}'.format(query_str))
    else:
        response_jsons_df = pd.DataFrame.from_records(response_jsons,
                                                      index='id')
        nodes = response_jsons_df[response_jsons_df['type'] == 'node']
        nodes = nodes[~nodes.index.duplicated(keep='first')]
        ways = response_jsons_df[response_jsons_df['type'] == 'way']
        ways = ways[~ways.index.duplicated(keep='first')]
        response_jsons_df = pd.concat([nodes, ways], axis=0)
        response_jsons_df.reset_index(inplace=True)
        response_jsons = response_jsons_df.to_dict(orient='records')
        if record_count - len(response_jsons) > 0:
            log('{:,} duplicate records removed. Took {:,.2f} seconds'.format(
                record_count - len(response_jsons), time.time() - start_time))

    return {'elements': response_jsons}