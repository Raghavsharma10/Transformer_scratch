def symbolic(self, mtx):
        """
        Perform symbolic object (symbolic LU decomposition) computation for a given
        sparsity pattern.
        """
        self.free_symbolic()

        indx = self._getIndx(mtx)

        if not assumeSortedIndices:
            # row/column indices cannot be assumed to be sorted
            mtx.sort_indices()

        if self.isReal:
            status, self._symbolic\
                    = self.funs.symbolic(mtx.shape[0], mtx.shape[1],
                                          mtx.indptr,
                                          indx,
                                          mtx.data,
                                          self.control, self.info)
        else:
            real, imag = mtx.data.real.copy(), mtx.data.imag.copy()
            status, self._symbolic\
                    = self.funs.symbolic(mtx.shape[0], mtx.shape[1],
                                          mtx.indptr,
                                          indx,
                                          real, imag,
                                          self.control, self.info)

        if status != UMFPACK_OK:
            raise RuntimeError('%s failed with %s' % (self.funs.symbolic,
                                                       umfStatus[status]))

        self.mtx = mtx