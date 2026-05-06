def tower_layout(graph, height='freeenergy', scale=None, center=None, dim=2):
    """
    Position all nodes of graph stacked on top of each other.

    Parameters
    ----------
    graph : `networkx.Graph` or `list` of nodes
        A position will be assigned to every node in graph.
    height : `str` or `None`, optional
        The node attribute that holds the numerical value used for the node
        height. This defaults to ``'freeenergy'``. If `None`, all node heights
        are set to zero.
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
    >>> from pyrrole import ChemicalSystem
    >>> from pyrrole.atoms import create_data, read_cclib
    >>> from pyrrole.drawing import tower_layout
    >>> data = create_data(
    ...     read_cclib("data/acetate/acetic_acid.out", "AcOH(g)"),
    ...     read_cclib("data/acetate/acetic_acid@water.out", "AcOH(aq)"))
    >>> digraph = (ChemicalSystem("AcOH(g) <=> AcOH(aq)", data)
    ...            .to_digraph())
    >>> layout = tower_layout(digraph)
    >>> layout['AcOH(g)']
    array([   0.        , -228.56450866])

    Passing ``scale=1`` means scaling positions to ``(-1, 1)`` in all axes:

    >>> layout = tower_layout(digraph, scale=1)
    >>> layout['AcOH(g)'][1] <= 1.
    True

    """
    # TODO: private function of packages should not be used.
    graph, center = _nx.drawing.layout._process_params(graph, center, dim)

    num_nodes = len(graph)
    if num_nodes == 0:
        return {}
    elif num_nodes == 1:
        return {_nx.utils.arbitrary_element(graph): center}

    paddims = max(0, (dim - 2))

    if height is None:
        y = _np.zeros(len(graph))
    else:
        y = _np.array([data for node, data in graph.nodes(data=height)])
    pos_arr = _np.column_stack([_np.zeros((num_nodes, 1)), y,
                                _np.zeros((num_nodes, paddims))])

    if scale is not None:
        pos_arr = _nx.drawing.layout.rescale_layout(pos_arr,
                                                    scale=scale) + center
    pos = dict(zip(graph, pos_arr))

    # TODO: make test
    return pos