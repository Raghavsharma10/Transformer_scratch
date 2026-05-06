def numeric(self, mtx):
        """
        Perform numeric object (LU decomposition) computation using the
        symbolic decomposition. The symbolic decomposition is (re)computed
        if necessary.
        """

        self.free_numeric()

        if self._symbolic is None:
            self.symbolic(mtx)

        indx = self._getIndx(mtx)

        failCount = 0
        while 1:
            if self.isReal:
                status, self._numeric\
                        = self.funs.numeric(mtx.indptr, indx, mtx.data,
                                             self._symbolic,
                                             self.control, self.info)
            else:
                real, imag = mtx.data.real.copy(), mtx.data.imag.copy()
                status, self._numeric\
                        = self.funs.numeric(mtx.indptr, indx,
                                             real, imag,
                                             self._symbolic,
                                             self.control, self.info)

            if status != UMFPACK_OK:
                if status == UMFPACK_WARNING_singular_matrix:
                    warnings.warn('Singular matrix', UmfpackWarning)
                    break
                elif status in (UMFPACK_ERROR_different_pattern,
                                UMFPACK_ERROR_invalid_Symbolic_object):
                    # Try again.
                    warnings.warn('Recomputing symbolic', UmfpackWarning)
                    self.symbolic(mtx)
                    failCount += 1
                else:
                    failCount += 100
            else:
                break
            if failCount >= 2:
                raise RuntimeError('%s failed with %s' % (self.funs.numeric,
                                                           umfStatus[status]))