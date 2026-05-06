def count_init(self):
        """Compute the counts at the first time step

        Returns
        -------
        n : ndarray(nstates)
            n[i] is the number of trajectories starting in state i

        """
        if self.hidden_state_trajectories is None:
            raise RuntimeError('HMM model does not have a hidden state trajectory.')

        n = [traj[0] for traj in self.hidden_state_trajectories]
        return np.bincount(n, minlength=self.nstates)