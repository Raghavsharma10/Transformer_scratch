def toset_from_tosets(*tosets):  # Note: a setlist is perfect representation of a toset as it's totally ordered and it's a set, i.e. a toset
    '''
    Create totally ordered set (toset) from tosets.

    These tosets, when merged, form a partially ordered set. The linear
    extension of this poset, a toset, is returned.

    .. warning:: untested

    Parameters
    ----------
    tosets : Iterable[~collections_extended.setlist]
        Tosets to merge.

    Raises
    ------
    ValueError
        If the tosets (derived from the lists) contradict each other. E.g. 
        ``[a, b]`` and ``[b, c, a]`` contradict each other.

    Returns
    -------
    toset : ~collectiontions_extended.setlist
        Totally ordered set.
    '''
    # Construct directed graph with: a <-- b iff a < b and adjacent in a list
    graph = nx.DiGraph()
    for toset in tosets:
        graph.add_nodes_from(toset)
        graph.add_edges_from(windowed(reversed(toset)))

    # No cycles allowed
    if not nx.is_directed_acyclic_graph(graph): #TODO could rely on NetworkXUnfeasible https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.algorithms.dag.topological_sort.html
        raise ValueError('Given tosets contradict each other')  # each cycle is a contradiction, e.g. a > b > c > a

    # Topological sort
    return setlist(nx.topological_sort(graph, reverse=True))