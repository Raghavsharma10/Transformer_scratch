def generate_gml(username, nodes, edges, cache=False):
    """
    Generate a GML format file representing the given graph attributes
    """

    # file segment that represents all the nodes in graph
    node_content = ""
    for i in range(len(nodes)):
        node_id = "\t\tid %d\n" % (i + 1)
        node_label = "\t\tlabel \"%s\"\n" % (nodes[i])

        node_content += format_node(node_id, node_label)

    # file segment that represents all the edges in graph
    edge_content = ""
    for i in range(len(edges)):
        edge = edges[i]

        edge_source = "\t\tsource %d\n" % (nodes.index(edge[0]) + 1)
        edge_target = "\t\ttarget %d\n" % (nodes.index(edge[1]) + 1)

        edge_content += format_edge(edge_source, edge_target)

    # formatted file content
    content = format_content(node_content, edge_content)

    with open(username_to_file(username), 'w') as f:
        f.write(content)

    # save the file for further use
    if cache:
    	cache_file(username_to_file(username))