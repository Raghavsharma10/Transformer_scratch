def Areml_eigh(self):
        """compute the eigenvalue decomposition of Astar"""
        s,U = LA.eigh(self.Areml(),lower=True)
        i_pos = (s>1e-10)
        s = s[i_pos]
        U = U[:,i_pos]
        return s,U