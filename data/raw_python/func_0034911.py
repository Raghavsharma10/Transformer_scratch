def getParams(self):
        """ get params """
        rv = np.array([])
        if self.n_terms>0:
            rv = np.concatenate([np.reshape(self.B[term_i],self.B[term_i].size, order='F') for term_i in range(self.n_terms)])
        return rv