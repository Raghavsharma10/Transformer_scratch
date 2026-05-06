def add_edges(self, from_idx, to_idx, weight=1, symmetric=False, copy=False):
    '''Adds all from->to edges. weight may be a scalar or 1d array.
    If symmetric=True, also adds to->from edges with the same weights.'''
    raise NotImplementedError()