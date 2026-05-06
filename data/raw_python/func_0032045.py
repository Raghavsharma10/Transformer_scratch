def draw_diagram_nodes(graph, pos=None, nodelist=None, node_size=.7,
                       node_color='k', style='solid', alpha=1.0, cmap=None,
                       vmin=None, vmax=None, ax=None, label=None):
    """
    Draw nodes of graph.

    This draws only the nodes of graph as horizontal lines at each
    ``y = pos[1]`` from ``x - node_size/2`` to ``x + node_size/2``, where
    ``x = pos[0]``.

    Parameters
    ----------
    graph : `networkx.Graph`
        A NetworkX graph.
    pos : mapping, optional
        A mapping with nodes as keys and positions as values. Positions should
        be sequences of length 2. If not specified (default), a diagram layout
        positioning will be computed. See `networkx.layout` and
        `pyrrole.drawing` for functions that compute node positions.
    nodelist : `list`, optional
        Draw only specified nodes (default is ``graph.nodes()``).
    node_size : scalar or array
        Size of nodes (default is ``.7``). If an array is specified it must be
        the same length as nodelist.
    node_color : color `str`, or array of `float`
        Node color. Can be a single color format `str` (default is ``'k'``), or
        a  sequence of colors with the same length as nodelist. If numeric
        values are specified they will be mapped to colors using the `cmap` and
        `vmin`, `vmax` parameters. See `matplotlib.hlines` for more details.
    style : `str` (``'solid'``, ``'dashed'``, ``'dotted'``, ``'dashdot'``)
        Edge line style (default is ``'solid'``). See `matplotlib.hlines` for
        more details.
    alpha : `float` or array of `float`, optional
        The node transparency. This can be a single alpha value (default is
        ``'1.0'``), in which case it will be applied to all the nodes of color.
        Otherwise, if it is an array, the elements of alpha will be applied to
        the colors in order (cycling through alpha multiple times if
        necessary).
    cmap : Matplotlib colormap, optional
        Colormap name or Colormap instance for mapping intensities of nodes.
    vmin : `float`, optional
        Minimum for node colormap scaling.
    vmax : `float`, optional
        Maximum for node colormap scaling.
    ax : `matplotlib.axes.Axes`, optional
        Draw the graph in the specified Matplotlib axes.
    label : `str`,  optional
        Label for legend.

    Returns
    -------
    `matplotlib.collections.LineCollection`
        `LineCollection` of the nodes.

    Raises
    ------
    networkx.NetworkXError
        Raised if a node has no position or one with bad value.

    Examples
    --------
    >>> import pandas as pd
    >>> from pyrrole import ChemicalSystem
    >>> from pyrrole.drawing import draw_diagram_nodes
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
    >>> nodes = draw_diagram_nodes(digraph)

    """
    if ax is None:
        ax = _plt.gca()

    if nodelist is None:
        nodelist = list(graph.nodes())

    if not nodelist or len(nodelist) == 0:  # empty nodelist, no drawing
        return None

    if pos is None:
        pos = diagram_layout(graph)

    try:
        xy = _np.asarray([pos[v] for v in nodelist])
    except KeyError as e:
        raise _nx.NetworkXError('Node {} has no position.'.format(e))
    except ValueError:
        raise _nx.NetworkXError('Bad value in node positions.')

    if isinstance(alpha, _collections.Iterable):
        node_color = _nx.drawing.apply_alpha(node_color, alpha, nodelist, cmap,
                                             vmin, vmax)
        alpha = None

    node_collection = ax.hlines(xy[:, 1],
                                xy[:, 0] - node_size/2.,
                                xy[:, 0] + node_size/2.,
                                colors=node_color,
                                linestyles=style,
                                label=label,
                                cmap=cmap)

    node_collection.set_zorder(2)
    return node_collection