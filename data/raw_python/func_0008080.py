def log_p_obs(self, obs, out=None, dtype=np.float32):
        """
        Returns the element-wise logarithm of the output probabilities for an entire trajectory and all hidden states

        This is a default implementation that will take the log of p_obs(obs) and should only be used if p_obs(obs)
        is numerically stable. If there is any danger of running into numerical problems *during* the calculation of
        p_obs, this function should be overwritten in order to compute the log-probabilities directly.

        Parameters
        ----------
        obs : ndarray((T), dtype=int)
            a discrete trajectory of length T

        Return
        ------
        p_o : ndarray (T,N)
            the log probability of generating the symbol at time point t from any of the N hidden states

        """
        if out is None:
            return np.log(self.p_obs(obs))
        else:
            self.p_obs(obs, out=out, dtype=dtype)
            np.log(out, out=out)
            return out