def linsolve(self, sys, mtx, rhs, autoTranspose=False):
        """
        One-shot solution of system of linear equation. Reuses Numeric object
        if possible.

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

        if self._numeric is None:
            self.numeric(mtx)
        else:
            if self.mtx is not mtx:
                self.numeric(mtx)

        sol = self.solve(sys, mtx, rhs, autoTranspose)
        self.free_numeric()

        return sol