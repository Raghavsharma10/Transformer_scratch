def count_matrix(self):
        # TODO: does this belong here or to the BHMM sampler, or in a subclass containing HMM with data?
        """Compute the transition count matrix from hidden state trajectory.

        Returns
        -------
        C : numpy.array with shape (nstates,nstates)
            C[i,j] is the number of transitions observed from state i to state j

        Raises
        ------
        RuntimeError
            A RuntimeError is raised if the HMM model does not yet have a hidden state trajectory associated with it.

        Examples
        --------

        """
        if self.hidden_state_trajectories is None:
            raise RuntimeError('HMM model does not have a hidden state trajectory.')

        C = msmest.count_matrix(self.hidden_state_trajectories, 1, nstates=self._nstates)
        return C.toarray()