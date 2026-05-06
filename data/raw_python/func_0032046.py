def draw_diagram_edges(graph, pos=None, edgelist=None, width=1.0,
                       edge_color='k', style='dashed', alpha=1.0,
                       edge_cmap=None, edge_vmin=None, edge_vmax=None, ax=None,
                       label=None, nodelist=None, node_size=.7):
    """
    Draw edges of graph.

    This draws only the edges of a graph.

    Parameters
    ----------
    graph : `networkx.Graph`
        A NetworkX graph.
    pos : mapping, optional
        A mapping with nodes as keys and positions as values. Positions should
        be sequences of length 2. If not specified (default), a diagram layout
        positioning will be computed. See `networkx.layout` and
        `pyrrole.drawing` for functions that compute node positions.
    edgelist : collection of edge `tuple`
        Draw only specified edges (default is ``graph.edges()``).
    width : `float`, or array of `float`
        Line width of edges (default is ``1.0``).
    edge_color : color `str`, or array of `float`
        Edge color. Can be a single color format `str` (default is ``'r'``),
        or a sequence of colors with the same length as edgelist. If numeric
        values are specified they will be mapped to colors using the
        `edge_cmap` and `edge_vmin`, `edge_vmax` parameters.
    style : `str` (``'solid'``, ``'dashed'``, ``'dotted'``, ``'dashdot'``)
        Edge line style (default is ``'dashed'``). See `matplotlib.hlines` for
        more details.
    alpha : `float`, optional
        The edge transparency (default is ``1.0``).
    edge_cmap : Matplotlib colormap, optional
        Colormap for mapping intensities of edges.
    edge_vmin : `float`, optional
        Minimum for edge colormap scaling.
    edge_vmax : `float`, optional
        Maximum for edge colormap scaling.
    ax : `matplotlib.axes.Axes`, optional
        Draw the graph in the specified Matplotlib axes.
    label : `str`,  optional
        Label for legend.
    nodelist : `list`, optional
        Draw only specified nodes (default is ``graph.nodes()``).
    node_size : scalar or array
        Size of nodes (default is ``.7``). If an array is specified it must be
        the same length as nodelist.

    Returns
    -------
    `matplotlib.collections.LineCollection`
        `LineCollection` of the edges.

    Raises
    ------
    networkx.NetworkXError
        Raised if a node has no position or one with bad value.
    ValueError
        Raised if `edge_color` contains something other than color names (one
        or a list of one per edge) or numbers.

    Examples
    --------
    >>> import pandas as pd
    >>> from pyrrole import ChemicalSystem
    >>> from pyrrole.drawing import draw_diagram_edges
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
    >>> edges = draw_diagram_edges(digraph)

    """
    if ax is None:
        ax = _plt.gca()

    if edgelist is None:
        edgelist = list(graph.edges())

    if not edgelist or len(edgelist) == 0:  # no edges!
        return None

    if nodelist is None:
        nodelist = list(graph.nodes())

    if pos is None:
        pos = diagram_layout(graph)

    try:
        # set edge positions
        edge_pos = _np.asarray([(pos[e[0]] + node_size/2.,
                                 pos[e[1]] - node_size/2.) for e in edgelist])
    except KeyError as e:
        raise _nx.NetworkXError('Node {} has no position.'.format(e))
    except ValueError:
        raise _nx.NetworkXError('Bad value in node positions.')

    if not _cb.iterable(width):
        lw = (width,)
    else:
        lw = width

    if not isinstance(edge_color, str) \
            and _cb.iterable(edge_color) \
            and len(edge_color) == len(edge_pos):
        if _np.alltrue([isinstance(c, str) for c in edge_color]):
            # (should check ALL elements)
            # list of color letters such as ['k','r','k',...]
            edge_colors = tuple([_colorConverter.to_rgba(c, alpha)
                                 for c in edge_color])
        elif _np.alltrue([not isinstance(c, str) for c in edge_color]):
            # If color specs are given as (rgb) or (rgba) tuples, we're OK
            if _np.alltrue([_cb.iterable(c) and len(c) in (3, 4)
                            for c in edge_color]):
                edge_colors = tuple(edge_color)
            else:
                # numbers (which are going to be mapped with a colormap)
                edge_colors = None
        else:
            raise ValueError('edge_color must contain color names or numbers')
    else:
        if isinstance(edge_color, str) or len(edge_color) == 1:
            edge_colors = (_colorConverter.to_rgba(edge_color, alpha), )
        else:
            raise ValueError('edge_color must be a color or list of one color '
                             ' per edge')

    edge_collection = _LineCollection(edge_pos,
                                      colors=edge_colors,
                                      linewidths=lw,
                                      antialiaseds=(1,),
                                      linestyle=style,
                                      transOffset=ax.transData)

    edge_collection.set_zorder(1)  # edges go behind nodes
    edge_collection.set_label(label)
    ax.add_collection(edge_collection)

    if _cb.is_numlike(alpha):
        edge_collection.set_alpha(alpha)

    if edge_colors is None:
        if edge_cmap is not None:
            assert(isinstance(edge_cmap, _Colormap))
        edge_collection.set_array(_np.asarray(edge_color))
        edge_collection.set_cmap(edge_cmap)
        if edge_vmin is not None or edge_vmax is not None:
            edge_collection.set_clim(edge_vmin, edge_vmax)
        else:
            edge_collection.autoscale()

    ax.autoscale_view()
    return edge_collection