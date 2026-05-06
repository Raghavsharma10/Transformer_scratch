def getResiduals(self):
        """ regress out fixed effects and results residuals """
        X = np.zeros((self.N*self.P,self.n_fixed_effs))
        ip = 0
        for i in range(self.n_terms):
            Ki = self.A[i].shape[0]*self.F[i].shape[1]
            X[:,ip:ip+Ki] = np.kron(self.A[i].T,self.F[i])
            ip += Ki
        y = np.reshape(self.Y,(self.Y.size,1),order='F')
        RV = regressOut(y,X)
        RV = np.reshape(RV,self.Y.shape,order='F')
        return RV