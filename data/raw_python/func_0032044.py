def diagram_layout(graph, height='freeenergy', sources=None, targets=None,
                   pos=None, scale=None, center=None, dim=2):
    """
    Position nodes such that paths are highlighted, from left to right.

    Parameters
    ----------
    graph : `networkx.Graph` or `list` of nodes
        A position will be assigned to every node in graph.
    height : `str` or `None`, optional
        The node attribute that holds the numerical value used for the node
        height. This defaults to ``'freeenergy'``. If `None`, all node heights
        are set to zero.
    sources : `list` of `str`
        All simple paths starting at members of `sources` are considered.
        Defaults to all nodes of graph.
    targets : `list` of `str`
        All simple paths ending at members of `targets` are considered.
        Defaults to all nodes of graph.
    pos : mapping, optional
        Initial positions for nodes as a mapping with node as keys and
        values as a coordinate `list` or `tuple`. If not specified (default),
        initial positions are computed with `tower_layout`.
    scale : number, optional
        Scale factor for positions.
    center : array-like, optional
        Coordinate pair around which to center the layout. Default is the
        origin.
    dim : `int`
        Dimension of layout. If `dim` > 2, the remaining dimensions are set to
        zero in the returned positions.

    Returns
    -------
    pos : mapping
        A mapping of positions keyed by node.

    Examples
    --------
    >>> import pandas as pd
    >>> from pyrrole import ChemicalSystem
    >>> from pyrrole.drawing import diagram_layout
    >>> data = pd.DataFrame(
    ...     [{"name": "Separated_Reactants", "freeenergy": 0.},
    ...      {"name": "mlC1", "freeenergy": -5.4},
    ...      {"name": "mlC2", "freeenergy": -15.6},
    ...      {"name": "mTS1", "freeenergy": 28.5, "color": "g"},
    ...      {"name": "mCARB1", "freeenergy": -9.7},
    ...      {"name": "mCARB2", "freeenergy": -19.8},
    ...      {"name": "mCARBX", "freeenergy": 20}]).set_index("name")
    >>> system = ChemicalSystem(
    ...     ["Separated_Reactants -> mlC1 -> mTS1",
    ...      "Separated_Reactants -> mlC2 -> mTS1",
    ...      "mCARB2 <- mTS1 -> mCARB1",
    ...      "Separated_Reactants -> mCARBX"], data)
    >>> digraph = system.to_digraph()
    >>> layout = diagram_layout(digraph)
    >>> layout['mCARB2']
    array([  3. , -19.8])

    Passing ``scale=1`` means scaling positions to ``(-1, 1)`` in all axes:

    >>> layout = diagram_layout(digraph, scale=1)
    >>> layout['mTS1'][1] <= 1.
    True

    """
    # TODO: private function of packages should not be used.
    graph, center = _nx.drawing.layout._process_params(graph, center, dim)

    num_nodes = len(graph)
    if num_nodes == 0:
        return {}
    elif num_nodes == 1:
        return {_nx.utils.arbitrary_element(graph): center}

    if sources is None:
        sources = graph.nodes()
    if targets is None:
        targets = graph.nodes()
    simple_paths = [path for source in set(sources) for target in set(targets)
                    for path in _nx.all_simple_paths(graph, source, target)]

    if pos is None:
        pos = tower_layout(graph, height=height, scale=None, center=center,
                           dim=dim)

    for path in simple_paths:
        for n, step in enumerate(path):
            if pos[step][0] < n:
                pos[step][0] = n

    if scale is not None:
        pos_arr = _np.array([pos[node] for node in graph])
        pos_arr = _nx.drawing.layout.rescale_layout(pos_arr,
                                                    scale=scale) + center
        pos = dict(zip(graph, pos_arr))

    # TODO: make test
    return pos