def solve_sparse(self, B):
        """
        Solve linear equation of the form A X = B. Where B and X are sparse matrices.

        Parameters
        ----------
        B : any scipy.sparse matrix
            Right-hand side of the matrix equation.
            Note: it will be converted to csc_matrix via `.tocsc()`.

        Returns
        -------
        X : csc_matrix
            Solution to the matrix equation as a csc_matrix
        """
        B = B.tocsc()
        cols = list()
        for j in xrange(B.shape[1]):
            col = self.solve(B[:,j])
            cols.append(csc_matrix(col))
        return hstack(cols)