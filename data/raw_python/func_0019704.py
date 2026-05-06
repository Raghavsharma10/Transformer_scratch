def lu(self, mtx):
        """
        Perform LU decomposition.

        For a given matrix A, the decomposition satisfies::

                LU = PRAQ        when do_recip is true
                LU = P(R^-1)AQ   when do_recip is false

        Parameters
        ----------
        mtx : scipy.sparse.csc_matrix or scipy.sparse.csr_matrix
            Input.

        Returns
        -------
        L : csr_matrix
            Lower triangular m-by-min(m,n) CSR matrix
        U : csc_matrix
            Upper triangular min(m,n)-by-n CSC matrix
        P : ndarray
            Vector of row permutations
        Q : ndarray
            Vector of column permutations
        R : ndarray
            Vector of diagonal row scalings
        do_recip : bool
            Whether R is R^-1 or R

        """

        # this should probably be changed
        mtx = mtx.tocsc()
        self.numeric(mtx)

        # first find out how much space to reserve
        (status, lnz, unz, n_row, n_col, nz_udiag)\
                 = self.funs.get_lunz(self._numeric)

        if status != UMFPACK_OK:
            raise RuntimeError('%s failed with %s' % (self.funs.get_lunz,
                                                       umfStatus[status]))

        # allocate storage for decomposition data
        i_type = mtx.indptr.dtype

        Lp = np.zeros((n_row+1,), dtype=i_type)
        Lj = np.zeros((lnz,), dtype=i_type)
        Lx = np.zeros((lnz,), dtype=np.double)

        Up = np.zeros((n_col+1,), dtype=i_type)
        Ui = np.zeros((unz,), dtype=i_type)
        Ux = np.zeros((unz,), dtype=np.double)

        P = np.zeros((n_row,), dtype=i_type)
        Q = np.zeros((n_col,), dtype=i_type)

        Dx = np.zeros((min(n_row,n_col),), dtype=np.double)

        Rs = np.zeros((n_row,), dtype=np.double)

        if self.isReal:
            (status,do_recip) = self.funs.get_numeric(Lp,Lj,Lx,Up,Ui,Ux,
                                                       P,Q,Dx,Rs,
                                                       self._numeric)

            if status != UMFPACK_OK:
                raise RuntimeError('%s failed with %s'
                        % (self.funs.get_numeric, umfStatus[status]))

            L = sp.csr_matrix((Lx,Lj,Lp),(n_row,min(n_row,n_col)))
            U = sp.csc_matrix((Ux,Ui,Up),(min(n_row,n_col),n_col))
            R = Rs

            return (L,U,P,Q,R,bool(do_recip))

        else:
            # allocate additional storage for imaginary parts
            Lz = np.zeros((lnz,), dtype=np.double)
            Uz = np.zeros((unz,), dtype=np.double)
            Dz = np.zeros((min(n_row,n_col),), dtype=np.double)

            (status,do_recip) = self.funs.get_numeric(Lp,Lj,Lx,Lz,Up,Ui,Ux,Uz,
                                                      P,Q,Dx,Dz,Rs,
                                                      self._numeric)

            if status != UMFPACK_OK:
                raise RuntimeError('%s failed with %s'
                        % (self.funs.get_numeric, umfStatus[status]))

            Lxz = np.zeros((lnz,), dtype=np.complex128)
            Uxz = np.zeros((unz,), dtype=np.complex128)
            Dxz = np.zeros((min(n_row,n_col),), dtype=np.complex128)

            Lxz.real,Lxz.imag = Lx,Lz
            Uxz.real,Uxz.imag = Ux,Uz
            Dxz.real,Dxz.imag = Dx,Dz

            L = sp.csr_matrix((Lxz,Lj,Lp),(n_row,min(n_row,n_col)))
            U = sp.csc_matrix((Uxz,Ui,Up),(min(n_row,n_col),n_col))
            R = Rs

            return (L,U,P,Q,R,bool(do_recip))