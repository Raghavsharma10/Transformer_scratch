def data_duration(self, data):
        '''Compute the valid data duration of a dict

        Parameters
        ----------
        data : dict
            As produced by pumpp.transform

        Returns
        -------
        length : int
            The minimum temporal extent of a dynamic observation in data
        '''
        # Find all the time-like indices of the data
        lengths = []
        for key in self._time:
            for idx in self._time.get(key, []):
                lengths.append(data[key].shape[idx])

        return min(lengths)