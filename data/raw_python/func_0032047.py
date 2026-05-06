def draw_diagram_labels(graph, pos=None, labels=None, font_size=12,
                        font_color='k', font_family='sans-serif',
                        font_weight='normal', alpha=1.0, bbox=None, ax=None,
                        offset=None, **kwds):
    """
    Draw node labels of graph.

    This draws only the node labels of a graph.

    Parameters
    ----------
    graph : `networkx.Graph`
        A NetworkX graph.
    pos : mapping, optional
        A mapping with nodes as keys and positions as values. Positions should
        be sequences of length 2. If not specified (default), a diagram layout
        positioning will be computed. See `networkx.layout` and
        `pyrrole.drawing` for functions that compute node positions.
    labels : mapping, optional
        Node labels in a mapping keyed by node of text labels.
    font_size : `int`, optional
       Font size for text labels (default is ``12``).
    font_color : `str`, optional
       Font color `str` (default is ``'k'``, i.e., black).
    font_family : `str`, optional
       Font family (default is ``'sans-serif'``).
    font_weight : `str`, optional
       Font weight (default is ``'normal'``).
    alpha : `float`, optional
        The text transparency (default is ``1.0``).
    ax : `matplotlib.axes.Axes`, optional
        Draw the graph in the specified Matplotlib axes.
    offset : array-like or `str`, optional
        Label positions are summed to this before drawing. Defaults to zero
        vector. If `str`, can be either ``'above'`` (equivalent to
        ``(0, 1.5)``) or ``'below'`` (equivalent to ``(0, -1.5)``).

    Returns
    -------
    mapping
        Mapping of labels keyed on the nodes.

    Examples
    --------
    >>> import pandas as pd
    >>> from pyrrole import ChemicalSystem
    >>> from pyrrole.drawing import draw_diagram_labels
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
    >>> edges = draw_diagram_labels(digraph, font_color='blue',
    ...                             offset="below")
    >>> labels = {k: "{:g}".format(v)
    ...           for k, v in digraph.nodes(data='freeenergy')}
    >>> edges = draw_diagram_labels(digraph, labels=labels,
    ...                             offset="above")

    """
    if ax is None:
        ax = _plt.gca()

    if labels is None:
        labels = dict((n, n) for n in graph.nodes())

    if pos is None:
        pos = diagram_layout(graph)

    if offset is None:
        offset = _np.array([0., 0.])
    elif offset == "above":
        offset = _np.array([0., 1.5])
    elif offset == "below":
        offset = _np.array([0., -1.5])

    # set optional alignment
    horizontalalignment = kwds.get('horizontalalignment', 'center')
    verticalalignment = kwds.get('verticalalignment', 'center')

    text_items = {}  # there is no text collection so we'll fake one
    for n, label in labels.items():
        (x, y) = _np.asanyarray(pos[n]) + _np.asanyarray(offset)
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same
        t = ax.text(x, y, label,
                    size=font_size,
                    color=font_color,
                    family=font_family,
                    weight=font_weight,
                    alpha=alpha,
                    horizontalalignment=horizontalalignment,
                    verticalalignment=verticalalignment,
                    transform=ax.transData,
                    bbox=bbox,
                    clip_on=True)
        text_items[n] = t

    return text_items