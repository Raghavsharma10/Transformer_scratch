def neighborhood_preserving_embedding(self, X, num_dims=None, reweight=True):
    '''Neighborhood Preserving Embedding (NPE, linearized LLE).'''
    if reweight:
      W = self.barycenter_edge_weights(X).matrix()
    else:
      W = self.matrix()
    # compute M = (I-W)'(I-W) as in LLE
    M = W.T.dot(W) - W.T - W
    if issparse(M):
      M = M.toarray()
    M.flat[::M.shape[0] + 1] += 1
    # solve generalized eig problem: X'MXa = \lambda X'Xa
    vals, vecs = eig(X.T.dot(M).dot(X), X.T.dot(X), overwrite_a=True,
                     overwrite_b=True)
    if num_dims is None:
      return vecs
    return vecs[:,:num_dims]