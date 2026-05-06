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

        if self.duration > duration:
            raise DataError('Data duration={} is less than '
                            'sample duration={}'.format(duration, self.duration))

        while True:
            # Generate a sampling interval
            yield self.rng.randint(0, duration - self.duration + 1)