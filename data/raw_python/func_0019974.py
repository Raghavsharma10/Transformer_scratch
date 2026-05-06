def kurt(self):
        """
        Kurtosis coefficient value as a result of an uncertainty calculation,
        defined as::
            
                      m4
            beta2 = ------
                    std**4

        where m4 is the fourth central moment and std is the standard deviation
        """
        mn = self.mean
        sd = self.std
        kt = 0.0 if abs(sd) <= 1e-8 else np.mean((self._mcpts - mn) ** 4) / sd ** 4
        return kt