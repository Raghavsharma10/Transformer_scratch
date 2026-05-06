def sample(self, data, interval):
        '''Sample a patch from the data object

        Parameters
        ----------
        data : dict
            A data dict as produced by pumpp.Pump.transform

        interval : slice
            The time interval to sample

        Returns
        -------
        data_slice : dict
            `data` restricted to `interval`.
        '''
        data_slice = dict()

        for key in data:
            if '_valid' in key:
                continue

            index = [slice(None)] * data[key].ndim

            # if we have multiple observations for this key, pick one
            index[0] = self.rng.randint(0, data[key].shape[0])
            index[0] = slice(index[0], index[0] + 1)

            for tdim in self._time[key]:
                index[tdim] = interval

            data_slice[key] = data[key][tuple(index)]

        return data_slice