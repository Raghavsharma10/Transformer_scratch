def matrix_from_edges(edges):
    '''Returns a sparse penalty matrix (D) from a list of edge pairs. Each edge
    can have an optional weight associated with it.'''
    max_col = 0
    cols = []
    rows = []
    vals = []
    if type(edges) is defaultdict:
        edge_list = []
        for i, neighbors in edges.items():
            for j in neighbors:
                if i <= j:
                    edge_list.append((i,j))
        edges = edge_list
    for i, edge in enumerate(edges):
        s, t = edge[0], edge[1]
        weight = 1 if len(edge) == 2 else edge[2]
        cols.append(min(s,t))
        cols.append(max(s,t))
        rows.append(i)
        rows.append(i)
        vals.append(weight)
        vals.append(-weight)
        if cols[-1] > max_col:
            max_col = cols[-1]
    return coo_matrix((vals, (rows, cols)), shape=(rows[-1]+1, max_col+1))