def locality_preserving_projections(self, coordinates, num_dims=None):
    '''Locality Preserving Projections (LPP, linearized Laplacian Eigenmaps).'''
    X = np.atleast_2d(coordinates)  # n x d
    L = self.laplacian(normed=True)  # n x n
    u,s,_ = np.linalg.svd(X.T.dot(X))
    Fplus = np.linalg.pinv(u * np.sqrt(s))  # d x d
    n, d = X.shape
    if n >= d:  # optimized order: F(X'LX)F'
      T = Fplus.dot(X.T.dot(L.dot(X))).dot(Fplus.T)
    else:  # optimized order: (FX')L(XF')
      T = Fplus.dot(X.T).dot(L.dot(X.dot(Fplus.T)))
    L = 0.5*(T+T.T)
    return _null_space(L, num_vecs=num_dims, overwrite=True)