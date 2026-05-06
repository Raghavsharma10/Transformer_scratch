def bicolor_spectral(self):
    '''Returns an approximate 2-coloring as an array of booleans.

    From "A Multiscale Pyramid Transform for Graph Signals" by Shuman et al.
    Note: Assumes a single connected component, and may fail otherwise.
    '''
    lap = self.laplacian().astype(float)
    vals, vecs = eigs(lap, k=1, which='LM')
    vec = vecs[:,0].real
    return vec > 0 if vec[0] > 0 else vec < 0