def coloured_network(network, setup, filename):
    """
    Plots a coloured (hyper-)graph to a dot file

    Parameters
    ----------
    network : object
        An object implementing a method `__plot__` which must return the `networkx.MultiDiGraph`_ instance to be coloured.
        Typically, it will be an instance of either :class:`caspo.core.graph.Graph`, :class:`caspo.core.logicalnetwork.LogicalNetwork`
        or :class:`caspo.core.logicalnetwork.LogicalNetworkList`

    setup : :class:`caspo.core.setup.Setup`
        Experimental setup to be coloured in the network


    .. _networkx.MultiDiGraph: https://networkx.readthedocs.io/en/stable/reference/classes.multidigraph.html#networkx.MultiDiGraph
    """
    NODES_ATTR = {
        'DEFAULT':   {'color': 'black', 'fillcolor': 'white', 'style': 'filled, bold', 'fontname': 'Helvetica', 'fontsize': 18, 'shape': 'ellipse'},
        'STIMULI':   {'color': 'olivedrab3', 'fillcolor': 'olivedrab3'},
        'INHIBITOR': {'color': 'orangered', 'fillcolor': 'orangered'},
        'READOUT':   {'color': 'lightblue', 'fillcolor': 'lightblue'},
        'INHOUT':    {'color': 'orangered', 'fillcolor': 'SkyBlue2', 'style': 'filled, bold, diagonals'},
        'GATE' :     {'fillcolor': 'black', 'fixedsize': True, 'width': 0.2, 'height': 0.2, 'label': '.'}
    }

    EDGES_ATTR = {
        'DEFAULT': {'dir': 'forward', 'penwidth': 2.5},
        1 : {'color': 'forestgreen', 'arrowhead': 'normal'},
        -1 : {'color': 'red', 'arrowhead': 'tee'}
    }

    graph = network.__plot__()

    for node in graph.nodes():
        _type = 'DEFAULT'
        for attr, value in NODES_ATTR[_type].items():
            graph.node[node][attr] = value

        if 'gate' in graph.node[node]:
            _type = 'GATE'
        elif node in setup.stimuli:
            _type = 'STIMULI'
        elif node in setup.readouts and node in setup.inhibitors:
            _type = 'INHOUT'
        elif node in setup.readouts:
            _type = 'READOUT'
        elif node in setup.inhibitors:
            _type = 'INHIBITOR'

        if _type != 'DEFAULT':
            for attr, value in NODES_ATTR[_type].items():
                graph.node[node][attr] = value

    for source, target in graph.edges():
        for k in graph.edge[source][target]:
            for attr, value in EDGES_ATTR['DEFAULT'].items():
                graph.edge[source][target][k][attr] = value

            for attr, value in EDGES_ATTR[graph.edge[source][target][k]['sign']].items():
                graph.edge[source][target][k][attr] = value

            if 'weight' in graph.edge[source][target][k]:
                graph.edge[source][target][k]['penwidth'] = 5 * graph.edge[source][target][k]['weight']

    write_dot(graph, filename)