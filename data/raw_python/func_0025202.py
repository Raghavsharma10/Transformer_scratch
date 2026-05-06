def model(self, v=None):
        "Returns the model of node v"
        if v is None:
            v = self.estopping
        hist = self.hist
        trace = self.trace(v)
        ins = None
        if self._base._probability_calibration is not None:
            node = hist[-1]
            node.normalize()
            X = np.array([x.full_array() for x in node.hy]).T
            y = np.array(self._base._y_klass.full_array())
            mask = np.ones(X.shape[0], dtype=np.bool)
            mask[np.array(self._base._mask_ts.index)] = False
            ins = self._base._probability_calibration().fit(X[mask], y[mask])
        if self._classifier:
            nclasses = self._labels.shape[0]
        else:
            nclasses = None
        m = Model(trace, hist, nvar=self._base._nvar,
                  classifier=self._classifier, labels=self._labels,
                  probability_calibration=ins, nclasses=nclasses)
        return m