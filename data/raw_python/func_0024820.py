def sampleset(self, factor_bbox=10.0, num=1000):
        """Return ``x`` array that samples the feature.

        Parameters
        ----------
        factor_bbox : float
            Factor for ``bounding_box`` calculations.

        num : int
            Number of points to generate.

        """
        w1, w2 = self.bounding_box(factor=factor_bbox)

        if self._n_models == 1:
            w = np.logspace(w1, w2, num)
        else:
            w = list(map(partial(np.logspace, num=num), w1, w2))

        return np.asarray(w)