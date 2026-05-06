def output(self):
        """Rank 3 array representing output time series. Axis 0 is time, 
        axis 1 ranges across output variables of a single simulation, axis 2 
        ranges across different simulation instances."""
        subts = [rms.output for rms in self._subsims]
        distaxis = subts[0].ndim - 1
        return DistTimeseries(subts, distaxis, self._node_labels())