def crop(self, data):
        '''Crop a data dictionary down to its common time

        Parameters
        ----------
        data : dict
            As produced by pumpp.transform

        Returns
        -------
        data_cropped : dict
            Like `data` but with all time-like axes truncated to the
            minimum common duration
        '''

        duration = self.data_duration(data)
        data_out = dict()
        for key in data:
            idx = [slice(None)] * data[key].ndim
            for tdim in self._time.get(key, []):
                idx[tdim] = slice(duration)
            data_out[key] = data[key][tuple(idx)]

        return data_out