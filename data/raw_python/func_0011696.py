def f(self, v, t):
        """Aburn2012 equations right hand side, noise free term
        Args: 
          v: (8,) array 
             state vector
          t: number
             scalar time
        Returns:
          (8,) array
        """
        ret = np.zeros(8)
        ret[0] = v[4]
        ret[4] = (self.He1*self.ke1*(self.g1*self.S(v[1]-v[2]) + self.u_mean) -
                  2*self.ke1*v[4] - self.ke1*self.ke1*v[0])

        ret[1] = v[5]
        ret[5] = (self.He2*self.ke2*(self.g2*self.S(v[0]) + self.p_mean) -
                  2*self.ke2*v[5] - self.ke2*self.ke2*v[1])

        ret[2] = v[6]
        ret[6] = (self.Hi*self.ki*self.g4*self.S(v[3]) - 2*self.ki*v[6] -
                  self.ki*self.ki*v[2])

        ret[3] = v[7]
        ret[7] = (self.He3*self.ke3*self.g3*self.S(v[1]-v[2]) -
                  2*self.ke3*v[7] - self.ke3*self.ke3*v[3])
        return ret