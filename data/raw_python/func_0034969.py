def laplacian_pca(self, coordinates, num_dims=None, beta=0.5):
    '''Graph-Laplacian PCA (CVPR 2013).
    coordinates : (n,d) array-like, assumed to be mean-centered.
    beta : float in [0,1], scales how much PCA/LapEig contributes.
    Returns an approximation of input coordinates, ala PCA.'''
    X = np.atleast_2d(coordinates)
    L = self.laplacian(normed=True)
    kernel = X.dot(X.T)
    kernel /= eigsh(kernel, k=1, which='LM', return_eigenvectors=False)
    L /= eigsh(L, k=1, which='LM', return_eigenvectors=False)
    W = (1-beta)*(np.identity(kernel.shape[0]) - kernel) + beta*L
    if num_dims is None:
      vals, vecs = np.linalg.eigh(W)
    else:
      vals, vecs = eigh(W, eigvals=(0, num_dims-1), overwrite_a=True)
    return X.T.dot(vecs).dot(vecs.T).T