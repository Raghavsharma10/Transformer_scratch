def predict(self):
        """ predict the value of the fixed effect """
        RV = np.zeros((self.N,self.P))
        for term_i in range(self.n_terms):
            RV+=np.dot(self.Fstar()[term_i],np.dot(self.B()[term_i],self.Astar()[term_i]))
        return RV