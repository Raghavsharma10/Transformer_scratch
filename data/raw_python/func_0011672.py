def merge(self, ts):
        """Merge another timeseries with this one
        Arguments:
          ts (Timeseries): The two timeseries being merged must have the
            same shape except for axis 0.
        Returns: 
          Resulting merged timeseries which can have duplicate time points.
        """
        if ts.shape[1:] != self.shape[1:]:
            raise ValueError('Timeseries to merge must have compatible shapes')
        indices = np.vstack((self.tspan, ts.tspan)).argsort()
        return np.vstack((self, ts))[indices]