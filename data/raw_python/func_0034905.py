def Zstar(self):
        """ predict the value of the fixed effect """
        RV = self.Ystar().copy()
        for term_i in range(self.n_terms):
            if self.identity_trick and self.A_identity[term_i]:
                RV-=np.dot(self.Fstar()[term_i],self.B_hat()[term_i])
            else:
                RV-=np.dot(self.Fstar()[term_i],np.dot(self.B_hat()[term_i],self.Astar()[term_i]))
        self.clear_cache('DLZ')
        return RV