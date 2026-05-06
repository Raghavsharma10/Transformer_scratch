def generate_synthetic_state_trajectory(self, nsteps, initial_Pi=None, start=None, stop=None, dtype=np.int32):
        """Generate a synthetic state trajectory.

        Parameters
        ----------
        nsteps : int
            Number of steps in the synthetic state trajectory to be generated.
        initial_Pi : np.array of shape (nstates,), optional, default=None
            The initial probability distribution, if samples are not to be taken from the intrinsic
            initial distribution.
        start : int
            starting state. Exclusive with initial_Pi
        stop : int
            stopping state. Trajectory will terminate when reaching the stopping state before length number of steps.
        dtype : numpy.dtype, optional, default=numpy.int32
            The numpy dtype to use to store the synthetic trajectory.

        Returns
        -------
        states : np.array of shape (nstates,) of dtype=np.int32
            The trajectory of hidden states, with each element in range(0,nstates).

        Examples
        --------

        Generate a synthetic state trajectory of a specified length.

        >>> from bhmm import testsystems
        >>> model = testsystems.dalton_model()
        >>> states = model.generate_synthetic_state_trajectory(nsteps=100)

        """
        # consistency check
        if initial_Pi is not None and start is not None:
            raise ValueError('Arguments initial_Pi and start are exclusive. Only set one of them.')

        # Generate first state sample.
        if start is None:
            if initial_Pi is not None:
                start = np.random.choice(range(self._nstates), size=1, p=initial_Pi)
            else:
                start = np.random.choice(range(self._nstates), size=1, p=self._Pi)

        # Generate and return trajectory
        from msmtools import generation as msmgen
        traj = msmgen.generate_traj(self.transition_matrix, nsteps, start=start, stop=stop, dt=1)
        return traj.astype(dtype)