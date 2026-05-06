def node_pairs(nodes, ways, waynodes, two_way=True):
    """
    Create a table of node pairs with the distances between them.

    Parameters
    ----------
    nodes : pandas.DataFrame
        Must have 'lat' and 'lon' columns.
    ways : pandas.DataFrame
        Table of way metadata.
    waynodes : pandas.DataFrame
        Table linking way IDs to node IDs. Way IDs should be in the index,
        with a column called 'node_ids'.
    two_way : bool, optional
        Whether the routes are two-way. If True, node pairs will only
        occur once. Default is True.

    Returns
    -------
    pairs : pandas.DataFrame
        Will have columns of 'from_id', 'to_id', and 'distance'.
        The index will be a MultiIndex of (from id, to id).
        The distance metric is in meters.

    """
    start_time = time.time()

    def pairwise(l):
        return zip(islice(l, 0, len(l)), islice(l, 1, None))
    intersections = intersection_nodes(waynodes)
    waymap = waynodes.groupby(level=0, sort=False)
    pairs = []

    for id, row in ways.iterrows():
        nodes_in_way = waymap.get_group(id).node_id.values
        nodes_in_way = [x for x in nodes_in_way if x in intersections]

        if len(nodes_in_way) < 2:
            # no nodes to connect in this way
            continue

        for from_node, to_node in pairwise(nodes_in_way):
            if from_node != to_node:
                fn = nodes.loc[from_node]
                tn = nodes.loc[to_node]

                distance = round(gcd(fn.lat, fn.lon, tn.lat, tn.lon), 6)

                col_dict = {'from_id': from_node,
                            'to_id': to_node,
                            'distance': distance}

                for tag in config.settings.keep_osm_tags:
                    try:
                        col_dict.update({tag: row[tag]})
                    except KeyError:
                        pass

                pairs.append(col_dict)

                if not two_way:

                    col_dict = {'from_id': to_node,
                                'to_id': from_node,
                                'distance': distance}

                    for tag in config.settings.keep_osm_tags:
                        try:
                            col_dict.update({tag: row[tag]})
                        except KeyError:
                            pass

                    pairs.append(col_dict)

    pairs = pd.DataFrame.from_records(pairs)
    if pairs.empty:
        raise Exception('Query resulted in no connected node pairs. Check '
                        'your query parameters or bounding box')
    else:
        pairs.index = pd.MultiIndex.from_arrays([pairs['from_id'].values,
                                                 pairs['to_id'].values])
        log('Edge node pairs completed. Took {:,.2f} seconds'
            .format(time.time()-start_time))

        return pairs