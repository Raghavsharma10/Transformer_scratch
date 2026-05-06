def eccentricity(self, directed=None, weighted=None):
    '''Maximum distance from each vertex to any other vertex.'''
    sp = self.shortest_path(directed=directed, weighted=weighted)
    return sp.max(axis=0)