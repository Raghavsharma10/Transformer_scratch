def locally_linear_embedding(self, num_dims=None):
    '''Locally Linear Embedding (LLE).
    Note: may need to call barycenter_edge_weights() before this!
    '''
    W = self.matrix()
    # compute M = (I-W)'(I-W)
    M = W.T.dot(W) - W.T - W
    if issparse(M):
      M = M.toarray()
    M.flat[::M.shape[0] + 1] += 1
    return _null_space(M, num_vecs=num_dims, overwrite=True)