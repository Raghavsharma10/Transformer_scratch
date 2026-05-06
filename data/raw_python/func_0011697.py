def G(self, v, t):
        """Aburn2012 equations right hand side, noise term
        Args: 
          v: (8,) array 
             state vector
          t: number
             scalar time
        Returns:
          (8,1) array
          Only one matrix column, meaning that in this example we are modelling
          the noise input to pyramidal and spiny populations as fully 
          correlated.  To simulate uncorrelated inputs instead, use an array of
          shape (8, 2) with the second noise element [5,1] instead of [5,0].
        """
        ret = np.zeros((8, 1))
        ret[4,0] = self.ke1 * self.He1 * self.u_sdev
        ret[5,0] = self.ke2 * self.He2 * self.p_sdev
        return ret