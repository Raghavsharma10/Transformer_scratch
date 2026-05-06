def request_heatmap_facet(field, hm_filter, hm_grid_level, hm_limit):
    """
    heatmap facet query builder
    :param field: map the query to this field.
    :param hm_filter: From what region to plot the heatmap. Defaults to q.geo or otherwise the world.
    :param hm_grid_level: To explicitly specify the grid level, e.g. to let a user ask for greater or courser
    resolution than the most recent request. Ignores a.hm.limit.
    :param hm_limit: Non-0 triggers heatmap/grid faceting. This number is a soft maximum on thenumber of
    cells it should have. There may be as few as 1/4th this number in return. Note that a.hm.gridLevel can effectively
    ignore this value. The response heatmap contains a counts grid that can be null or contain null rows when all its
    values would be 0. See Solr docs for more details on the response format.
    :return:
    """

    if not hm_filter:
        hm_filter = '[-90,-180 TO 90,180]'

    params = {
        'facet': 'on',
        'facet.heatmap': field,
        'facet.heatmap.geom': hm_filter
    }

    if hm_grid_level:
        # note: aHmLimit is ignored in this case
        params['facet.heatmap.gridLevel'] = hm_grid_level
    else:
        # Calculate distErr that will approximate aHmLimit many cells as an upper bound
        rectangle = parse_geo_box(hm_filter)
        degrees_side_length = rectangle.length / 2
        cell_side_length = math.sqrt(float(hm_limit))
        cell_side_length_degrees = degrees_side_length / cell_side_length * 2
        params['facet.heatmap.distErr'] = str(float(cell_side_length_degrees))
        # TODO: not sure about if returning correct param values.

    # get_params = urllib.urlencode(params)
    return params