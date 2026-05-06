def _reshape_timeseries(self, ts):
        """Introduce a new axis 2 that ranges across nodes of the network"""
        if np.count_nonzero(np.diff(self._sublengths)) == 0:
            # then all submodels have the same dimension, so can reshape array
            # in place without copying data:
            subdim = self.dimension // self._n
            shp = list(ts.shape)
            shp[1] = self._n
            shp.insert(2, subdim)
            ts = ts.reshape(tuple(shp)).swapaxes(1, 2)
            # label variables only if all sub-models agree on the labels:
            all_var_labels = [m.labels for m in self.submodels]
            var_labels = all_var_labels[0]
            if all(v == var_labels for v in all_var_labels[1:]):
                ts.labels[1] = var_labels
            ts.labels[2] = self._node_labels()
            return ts
        else:
            # will pad with zeros for submodels with less variables
            subdim = max(self._sublengths)
            shp = list(ts.shape)
            shp[1] = subdim
            shp.insert(2, self._n)
            ar = np.zeros(shp)
            labels = ts.labels
            labels.insert(2, self._node_labels())
            for k in range(self._n):
                sl = slice(self._si[k], self._si[k+1])
                ar[:,:,k,...] = ts[:,sl,...]
            return Timeseries(ar, ts.tspan, labels)