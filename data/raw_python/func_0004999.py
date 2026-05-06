def indices(self, data):
        '''Generate patch indices

        Parameters
        ----------
        data : dict of np.ndarray
            As produced by pumpp.transform

        Yields
        ------
        start : int >= 0
            The start index of a sample patch
        '''
        duration = self.data_duration(data)

        while True:
            # Generate a sampling interval
            yield self.rng.randint(0, duration - self.min_duration + 1)