def directed_laplacian(self, D=None, eta=0.99, tol=1e-12, max_iter=500):
    '''Computes the directed combinatorial graph laplacian.
    http://www-all.cs.umass.edu/pubs/2007/johns_m_ICML07.pdf

    D: (optional) N-array of degrees
    eta: probability of not teleporting (see the paper)
    tol, max_iter: convergence params for Perron vector calculation
    '''
    W = self.matrix('dense')
    n = W.shape[0]
    if D is None:
      D = W.sum(axis=1)
    # compute probability transition matrix
    with np.errstate(invalid='ignore', divide='ignore'):
      P = W.astype(float) / D[:,None]
    P[D==0] = 0
    # start at the uniform distribution Perron vector (phi)
    old_phi = np.ones(n) / n
    # iterate to the fixed point (teleporting random walk)
    for _ in range(max_iter):
      phi = eta * old_phi.dot(P) + (1-eta)/n
      if np.abs(phi - old_phi).max() < tol:
        break
      old_phi = phi
    else:
      warnings.warn("phi failed to converge after %d iterations" % max_iter)
    # L = Phi - (Phi P + P' Phi)/2
    return np.diag(phi) - ((phi * P.T).T + P.T * phi)/2