def output(self):
        """Rank 3 array representing output time series. Axis 0 is time, 
        axis 1 ranges across output variables of a single simulation, 
        axis 2 ranges across different simulation instances."""
        subts = [s.output for s in self.sims]
        sub_ndim = subts[0].ndim
        if sub_ndim is 1:
            subts = [distob.expand_dims(ts, 1) for ts in subts]
            sub_ndim += 1
        nodeaxis = sub_ndim
        subts = [distob.expand_dims(ts, nodeaxis) for ts in subts]
        ts = subts[0].concatenate(subts[1:], axis=nodeaxis)
        ts.labels[nodeaxis] = self._node_labels()
        return ts