def p_obs(self, obs, out=None):
        """
        Returns the output probabilities for an entire trajectory and all hidden states

        Parameters
        ----------
        obs : ndarray((T), dtype=int)
            a discrete trajectory of length T

        Return
        ------
        p_o : ndarray (T,N)
            the probability of generating the symbol at time point t from any of the N hidden states

        """
        if out is None:
            out = self._output_probabilities[:, obs].T
            # out /= np.sum(out, axis=1)[:,None]
            return self._handle_outliers(out)
        else:
            if obs.shape[0] == out.shape[0]:
                np.copyto(out, self._output_probabilities[:, obs].T)
            elif obs.shape[0] < out.shape[0]:
                out[:obs.shape[0], :] = self._output_probabilities[:, obs].T
            else:
                raise ValueError('output array out is too small: '+str(out.shape[0])+' < '+str(obs.shape[0]))
            # out /= np.sum(out, axis=1)[:,None]
            return self._handle_outliers(out)