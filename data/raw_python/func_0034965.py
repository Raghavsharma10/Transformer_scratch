def laplacian_eigenmaps(self, num_dims=None, normed=True, val_thresh=1e-8):
    '''Laplacian Eigenmaps embedding.

    num_dims : dimension of embedded coordinates, defaults to input dimension
    normed : used for .laplacian() calculation
    val_thresh : threshold for omitting vectors with near-zero eigenvalues
    '''
    L = self.laplacian(normed=normed)
    return _null_space(L, num_dims, val_thresh, overwrite=True)