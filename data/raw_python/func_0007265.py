def intersection_nodes(waynodes):
    """
    Returns a set of all the nodes that appear in 2 or more ways.

    Parameters
    ----------
    waynodes : pandas.DataFrame
        Mapping of way IDs to node IDs as returned by `ways_in_bbox`.

    Returns
    -------
    intersections : set
        Node IDs that appear in 2 or more ways.

    """
    counts = waynodes.node_id.value_counts()
    return set(counts[counts > 1].index.values)