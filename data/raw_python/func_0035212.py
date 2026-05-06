def XanyKX(self):
        """
        compute cross covariance for any and rest
        """
        result = np.empty((self.P,self.F_any.shape[1],self.dof), order='C')
        #This is trivially parallelizable:
        for p in range(self.P):
            FanyD = self.Fstar_any * self.D[:,p:p+1]
            start = 0
            #This is trivially parallelizable:
            for term in range(self.len):
                stop = start + self.F[term].shape[1]*self.A[term].shape[0]
                result[p,:,start:stop] = self.XanyKX2_single_p_single_term(p=p, F1=FanyD, F2=self.Fstar[term], A2=self.Astar[term])
                start = stop
        return result