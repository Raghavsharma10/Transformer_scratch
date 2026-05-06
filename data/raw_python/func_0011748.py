def _reshape_output(self, ts):
        """Introduce a new axis 2 that ranges across nodes of the network"""
        subodim = len(self.submodels[0].output_vars)
        shp = list(ts.shape)
        shp[1] = subodim
        shp.insert(2, self._n)
        ts = ts.reshape(tuple(shp))
        ts.labels[2] = self._node_labels()
        return ts