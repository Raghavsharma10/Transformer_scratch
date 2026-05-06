def minimum_spanning_subtree(self):
    '''Returns the (undirected) minimum spanning tree subgraph.'''
    dist = self.matrix('dense', copy=True)
    dist[dist==0] = np.inf
    np.fill_diagonal(dist, 0)
    mst = ssc.minimum_spanning_tree(dist)
    return self.__class__.from_adj_matrix(mst + mst.T)