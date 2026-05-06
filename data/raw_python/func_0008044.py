def _forward_backward(self, itraj):
        """
        Estimation step: Runs the forward-back algorithm on trajectory with index itraj

        Parameters
        ----------
        itraj : int
            index of the observation trajectory to process

        Results
        -------
        logprob : float
            The probability to observe the observation sequence given the HMM
            parameters
        gamma : ndarray(T,N, dtype=float)
            state probabilities for each t
        count_matrix : ndarray(N,N, dtype=float)
            the Baum-Welch transition count matrix from the hidden state
            trajectory

        """
        # get parameters
        A = self._hmm.transition_matrix
        pi = self._hmm.initial_distribution
        obs = self._observations[itraj]
        T = len(obs)
        # compute output probability matrix
        # t1 = time.time()
        self._hmm.output_model.p_obs(obs, out=self._pobs)
        # t2 = time.time()
        # self._fbtimings[0] += t2-t1
        # forward variables
        logprob = hidden.forward(A, self._pobs, pi, T=T, alpha_out=self._alpha)[0]
        # t3 = time.time()
        # self._fbtimings[1] += t3-t2
        # backward variables
        hidden.backward(A, self._pobs, T=T, beta_out=self._beta)
        # t4 = time.time()
        # self._fbtimings[2] += t4-t3
        # gamma
        hidden.state_probabilities(self._alpha, self._beta, T=T, gamma_out=self._gammas[itraj])
        # t5 = time.time()
        # self._fbtimings[3] += t5-t4
        # count matrix
        hidden.transition_counts(self._alpha, self._beta, A, self._pobs, T=T, out=self._Cs[itraj])
        # t6 = time.time()
        # self._fbtimings[4] += t6-t5
        # return results
        return logprob