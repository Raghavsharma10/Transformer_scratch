def XKX(self):
        """
        compute self covariance for rest
        """
        cov_beta = np.zeros((self.dof,self.dof))
        start_row = 0
        #This is trivially parallelizable:
        for term1 in range(self.len):
            stop_row = start_row + self.A[term1].shape[0] * self.F[term1].shape[1]
            start_col = start_row
            #This is trivially parallelizable:
            for term2 in range(term1,self.len):
                stop_col = start_col + self.A[term2].shape[0] * self.F[term2].shape[1]
                cov_beta[start_row:stop_row, start_col:stop_col] = compute_X1KX2(Y=self.Ystar(), D=self.D, X1=self.Fstar[term1], X2=self.Fstar[term2], A1=self.Astar[term1], A2=self.Astar[term2])
                if term1!=term2:
                    cov_beta[start_col:stop_col, start_row:stop_row] = cov_beta[n_weights1:stop_row, n_weights2:stop_col].T
                start_col = stop_col
            start_row = stop_row
        return cov_beta