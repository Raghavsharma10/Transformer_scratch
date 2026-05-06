def create_plateaus(data, edges, plateau_size, plateau_vals, plateaus=None):
    '''Creates plateaus of constant value in the data.'''
    nodes = set(edges.keys())
    if plateaus is None:
        plateaus = []
        for i in range(len(plateau_vals)):
            if len(nodes) == 0:
                break
            node = np.random.choice(list(nodes))
            nodes.remove(node)
            plateau = [node]
            available = set(edges[node]) & nodes
            while len(nodes) > 0 and len(available) > 0 and len(plateau) < plateau_size:
                node = np.random.choice(list(available))
                plateau.append(node)
                available |= nodes & set(edges[node])
                available.remove(node)
            nodes -= set(plateau)
            plateaus.append(set(plateau))
    for p,v in zip(plateaus, plateau_vals):
        data[np.array(list(p), dtype=int)] = v
    return plateaus