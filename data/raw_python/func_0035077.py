def remove_edges(self, from_idx, to_idx, symmetric=False, copy=False):
    '''Removes all from->to and to->from edges.
    Note: the symmetric kwarg is unused.'''
    flat_inds = self._pairs.dot((self._num_vertices, 1))
    # convert to sorted order and flatten
    to_remove = (np.minimum(from_idx, to_idx) * self._num_vertices
                 + np.maximum(from_idx, to_idx))
    mask = np.in1d(flat_inds, to_remove, invert=True)
    res = self.copy() if copy else self
    res._pairs = res._pairs[mask]
    res._offdiag_mask = res._offdiag_mask[mask]
    return res