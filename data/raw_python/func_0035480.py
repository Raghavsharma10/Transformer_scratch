def betweenness(self, kind='vertex', directed=None, weighted=None):
    '''Computes the betweenness centrality of a graph.
    kind : string, either 'vertex' (default) or 'edge'
    directed : bool, defaults to self.is_directed()
    weighted : bool, defaults to self.is_weighted()
    '''
    assert kind in ('vertex', 'edge'), 'Invalid kind argument: ' + kind
    weighted = weighted is not False and self.is_weighted()
    directed = directed if directed is not None else self.is_directed()
    adj = self.matrix('csr')
    btw = betweenness(adj, weighted, kind=='vertex')
    # normalize if undirected
    if not directed:
      btw /= 2.
    return btw