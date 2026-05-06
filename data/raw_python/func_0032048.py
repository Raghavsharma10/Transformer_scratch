def draw_diagram(graph, pos=None, with_labels=True, offset=None, **kwds):
    """
    Draw a diagram for graph using Matplotlib.

    Draw graph as a simple energy diagram with Matplotlib with options for node
    positions, labeling, titles, and many other drawing features. See examples
    below.

    Parameters
    ----------
    graph : `networkx.Graph`
        A NetworkX graph.
    pos : mapping, optional
        A mapping with nodes as keys and positions as values. Positions should
        be sequences of length 2. If not specified (default) a diagram layout
        positioning will be computed. See `networkx.drawing.layout` and
        `pyrrole.drawing` for functions that compute node positions.
    with_labels : `bool`, optional
       Set to `True` (default) to draw labels on the nodes.
    offset : array-like or `str`, optional
        Label positions are summed to this before drawing. Defaults to
        ``'below'``. See `draw_diagram_labels` for more.

    Notes
    -----
    Further keywords are passed to `draw_diagram_nodes` and
    `draw_diagram_edges`. If `pos` is `None`, `diagram_layout` is also called
    and have keywords passed as well. The same happens with
    `draw_diagram_labels` if `with_labels` is `True`.

    Examples
    --------
    >>> import pandas as pd
    >>> from pyrrole import ChemicalSystem
    >>> from pyrrole.drawing import draw_diagram, draw_diagram_labels
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
    >>> draw_diagram(digraph)
    >>> labels = {k: "{:g}".format(v)
    ...           for k, v in digraph.nodes(data='freeenergy')}
    >>> edges = draw_diagram_labels(digraph, labels=labels,
    ...                             offset="above")

    """
    if pos is None:
        pos = diagram_layout(graph, **kwds)  # default to diagram layout

    node_collection = draw_diagram_nodes(graph, pos, **kwds)  # noqa
    edge_collection = draw_diagram_edges(graph, pos, **kwds)  # noqa
    if with_labels:
        if offset is None:
            # TODO: This changes the default behaviour of draw_diagram_labels.
            offset = "below"
        draw_diagram_labels(graph, pos, offset=offset, **kwds)
    _plt.draw_if_interactive()