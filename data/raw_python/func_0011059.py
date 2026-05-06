def __remove_equalities(self, equalities, momentequalities):
        """Attempt to remove equalities by solving the linear equations.
        """
        A = self.__process_equalities(equalities, momentequalities)
        if min(A.shape != np.linalg.matrix_rank(A)):
            print("Warning: equality constraints are linearly dependent! "
                  "Results might be incorrect.", file=sys.stderr)
        if A.shape[0] == 0:
            return
        c = np.array(self.obj_facvar)
        if self.verbose > 0:
            print("QR decomposition...")
        Q, R = np.linalg.qr(A[:, 1:].T, mode='complete')
        n = np.max(np.nonzero(np.sum(np.abs(R), axis=1) > 0)) + 1
        x = np.dot(Q[:, :n], np.linalg.solve(np.transpose(R[:n, :]), -A[:, 0]))
        self._new_basis = lil_matrix(Q[:, n:])
        # Transforming the objective function
        self._original_obj_facvar = self.obj_facvar
        self._original_constant_term = self.constant_term
        self.obj_facvar = self._new_basis.T.dot(c)
        self.constant_term += c.dot(x)
        x = np.append(1, x)
        # Transforming the moment matrix and localizing matrices
        new_F = lil_matrix((self.F.shape[0], self._new_basis.shape[1] + 1))
        new_F[:, 0] = self.F[:, :self.n_vars+1].dot(x).reshape((new_F.shape[0],
                                                                1))
        new_F[:, 1:] = self.F[:, 1:self.n_vars+1].\
            dot(self._new_basis)
        self._original_F = self.F
        self.F = new_F
        self.n_vars = self._new_basis.shape[1]
        if self.verbose > 0:
            print("Number of variables after solving the linear equations: %d"
                  % self.n_vars)