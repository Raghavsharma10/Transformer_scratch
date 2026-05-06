def ave_laplacian(self):
    '''Another kind of laplacian normalization, used in the matlab PVF code.
    Uses the formula: L = I - D^{-1} * W'''
    W = self.matrix('dense')
    # calculate -inv(D)
    Dinv = W.sum(axis=0)
    mask = Dinv!=0
    Dinv[mask] = -1./Dinv[mask]
    # calculate -inv(D) * W
    lap = (Dinv * W.T).T
    # add I
    lap.flat[::W.shape[0]+1] += 1
    # symmetrize
    return (lap + lap.T) / 2.0