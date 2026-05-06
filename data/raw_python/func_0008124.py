def _sampleHiddenStateTrajectory(self, obs, dtype=np.int32):
        """Sample a hidden state trajectory from the conditional distribution P(s | T, E, o)

        Parameters
        ----------
        o_t : numpy.array with dimensions (T,)
            observation[n] is the nth observation
        dtype : numpy.dtype, optional, default=numpy.int32
            The dtype to to use for returned state trajectory.

        Returns
        -------
        s_t : numpy.array with dimensions (T,) of type `dtype`
            Hidden state trajectory, with s_t[t] the hidden state corresponding to observation o_t[t]

        Examples
        --------
        >>> import bhmm
        >>> [model, observations, states, sampled_model] = bhmm.testsystems.generate_random_bhmm(ntrajectories=5, length=1000)
        >>> o_t = observations[0]
        >>> s_t = sampled_model._sampleHiddenStateTrajectory(o_t)

        """

        # Determine observation trajectory length
        T = obs.shape[0]

        # Convenience access.
        A = self.model.transition_matrix
        pi = self.model.initial_distribution

        # compute output probability matrix
        self.model.output_model.p_obs(obs, out=self.pobs)
        # compute forward variables
        logprob = hidden.forward(A, self.pobs, pi, T=T, alpha_out=self.alpha)[0]
        # sample path
        S = hidden.sample_path(self.alpha, A, self.pobs, T=T)

        return S