def isomap(self, num_dims=None, directed=None):
    '''Isomap embedding.

    num_dims : dimension of embedded coordinates, defaults to input dimension
    directed : used for .shortest_path() calculation
    '''
    W = -0.5 * self.shortest_path(directed=directed) ** 2
    kpca = KernelPCA(n_components=num_dims, kernel='precomputed')
    return kpca.fit_transform(W)