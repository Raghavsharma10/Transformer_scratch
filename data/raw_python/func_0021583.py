def hypercube_edges(dims, use_map=False):
    '''Create edge lists for an arbitrary hypercube. TODO: this is probably not the fastest way.'''
    edges = []
    nodes = np.arange(np.product(dims)).reshape(dims)
    for i,d in enumerate(dims):
        for j in range(d-1):
            for n1, n2 in zip(np.take(nodes, [j], axis=i).flatten(), np.take(nodes,[j+1], axis=i).flatten()):
                edges.append((n1,n2))
    if use_map:
        return edge_map_from_edge_list(edges)
    return edges