def compute_viterbi_paths(self):
        """
        Computes the viterbi paths using the current HMM model

        """
        # get parameters
        K = len(self._observations)
        A = self._hmm.transition_matrix
        pi = self._hmm.initial_distribution

        # compute viterbi path for each trajectory
        paths = np.empty(K, dtype=object)
        for itraj in range(K):
            obs = self._observations[itraj]
            # compute output probability matrix
            pobs = self._hmm.output_model.p_obs(obs)
            # hidden path
            paths[itraj] = hidden.viterbi(A, pobs, pi)

        # done
        return paths