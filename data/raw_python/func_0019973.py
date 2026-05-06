def skew(self):
        r"""
        Skewness coefficient value as a result of an uncertainty calculation,
        defined as::
            
              _____     m3
            \/beta1 = ------
                      std**3
        
        where m3 is the third central moment and std is the standard deviation
        """
        mn = self.mean
        sd = self.std
        sk = 0.0 if abs(sd) <= 1e-8 else np.mean((self._mcpts - mn) ** 3) / sd ** 3
        return sk