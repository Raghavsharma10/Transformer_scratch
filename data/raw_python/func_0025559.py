def nclasses(self, v):
        "Number of classes of v, also sets the labes"
        if not self.classifier:
            return 0
        if isinstance(v, list):
            self._labels = np.arange(len(v))
            return
        if not isinstance(v, np.ndarray):
            v = tonparray(v)
        self._labels = np.unique(v)
        return self._labels.shape[0]