def solve(self, sys, mtx, rhs, autoTranspose=False):
        """
        Solution of system of linear equation using the Numeric object.

        Parameters
        ----------
        sys : constant
            one of UMFPACK system description constants, like
            UMFPACK_A, UMFPACK_At, see umfSys list and UMFPACK docs
        mtx : scipy.sparse.csc_matrix or scipy.sparse.csr_matrix
            Input.
        rhs : ndarray
            Right Hand Side
        autoTranspose : bool
            Automatically changes `sys` to the transposed type, if `mtx` is in CSR,
            since UMFPACK assumes CSC internally

        Returns
        -------
        sol : ndarray
            Solution to the equation system.

        """
        if sys not in umfSys:
            raise ValueError('sys must be in' % umfSys)

        if autoTranspose and self.isCSR:
            ##
            # UMFPACK uses CSC internally...
            if self.family in umfRealTypes:
                ii = 0
            else:
                ii = 1
            if sys in umfSys_transposeMap[ii]:
                sys = umfSys_transposeMap[ii][sys]
            else:
                raise RuntimeError('autoTranspose ambiguous, switch it off')

        if self._numeric is not None:
            if self.mtx is not mtx:
                raise ValueError('must be called with same matrix as numeric()')
        else:
            raise RuntimeError('numeric() not called')

        indx = self._getIndx(mtx)

        if self.isReal:
            rhs = rhs.astype(np.float64)
            sol = np.zeros((mtx.shape[1],), dtype=np.float64)
            status = self.funs.solve(sys, mtx.indptr, indx, mtx.data, sol, rhs,
                                      self._numeric, self.control, self.info)
        else:
            rhs = rhs.astype(np.complex128)
            sol = np.zeros((mtx.shape[1],), dtype=np.complex128)
            mreal, mimag = mtx.data.real.copy(), mtx.data.imag.copy()
            sreal, simag = sol.real.copy(), sol.imag.copy()
            rreal, rimag = rhs.real.copy(), rhs.imag.copy()
            status = self.funs.solve(sys, mtx.indptr, indx,
                                      mreal, mimag, sreal, simag, rreal, rimag,
                                      self._numeric, self.control, self.info)
            sol.real, sol.imag = sreal, simag

        # self.funs.report_info( self.control, self.info )
        # pause()
        if status != UMFPACK_OK:
            if status == UMFPACK_WARNING_singular_matrix:
                ## Change inf, nan to zeros.
                warnings.warn('Zeroing nan and inf entries...', UmfpackWarning)
                sol[~np.isfinite(sol)] = 0.0
            else:
                raise RuntimeError('%s failed with %s' % (self.funs.solve,
                                                           umfStatus[status]))
        econd = 1.0 / self.info[UMFPACK_RCOND]
        if econd > self.maxCond:
            msg = '(almost) singular matrix! '\
                  + '(estimated cond. number: %.2e)' % econd
            warnings.warn(msg, UmfpackWarning)

        return sol