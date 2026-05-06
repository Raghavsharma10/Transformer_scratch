def symmetrize(self, method=None, copy=False):
    '''Symmetrizes (ignores method). Returns a copy if copy=True.'''
    if copy:
      return SymmEdgePairGraph(self._pairs.copy(),
                               num_vertices=self._num_vertices)
    shape = (self._num_vertices, self._num_vertices)
    flat_inds = np.union1d(np.ravel_multi_index(self._pairs.T, shape),
                           np.ravel_multi_index(self._pairs.T[::-1], shape))
    self._pairs = np.transpose(np.unravel_index(flat_inds, shape))
    return self