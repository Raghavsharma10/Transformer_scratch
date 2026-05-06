def setParams(self,params):
        """ set params """
        start = 0
        for i in range(self.n_terms):
            n_effects = self.B[i].size
            self.B[i] = np.reshape(params[start:start+n_effects],self.B[i].shape, order='F')
            start += n_effects