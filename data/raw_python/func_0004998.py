def indices(self, data):
        '''Generate patch start indices

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

        for start in range(0, duration - self.duration, self.stride):
            yield start