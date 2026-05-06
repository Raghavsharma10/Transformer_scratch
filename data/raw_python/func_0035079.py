def add_self_edges(self, weight=None, copy=False):
    '''Adds all i->i edges. weight may be a scalar or 1d array.'''
    ii = np.arange(self.num_vertices())
    return self.add_edges(ii, ii, weight=weight, symmetric=False, copy=copy)