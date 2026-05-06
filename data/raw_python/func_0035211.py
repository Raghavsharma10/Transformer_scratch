def XanyKXany(self):
        """
        compute self covariance for any
        """
        result = np.empty((self.P,self.F_any.shape[1],self.F_any.shape[1]), order='C')
        for p in range(self.P):
            X1D = self.Fstar_any * self.D[:,p:p+1]
            X1X2 = X1D.T.dot(self.Fstar_any)
            result[p] = X1X2
        return result