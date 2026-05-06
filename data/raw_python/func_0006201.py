def graph_from_dot_file(path):
    """Load graph as defined by a DOT file.

    The file is assumed to be in DOT format. It will
    be loaded, parsed and a Dot class will be returned,
    representing the graph.
    """

    fd = open(path, 'rb')
    data = fd.read()
    fd.close()

    return graph_from_dot_data(data)