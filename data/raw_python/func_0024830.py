def sampleset(self, factor_step=0.05, **kwargs):
        """Return ``x`` array that samples the feature.

        Parameters
        ----------
        factor_step : float
            Factor for sample step calculation. The step is calculated
            using ``factor_step * self.fwhm``.

        kwargs : dict
            Keyword(s) for ``bounding_box`` calculation.

        """
        w1, w2 = self.bounding_box(**kwargs)
        dw = factor_step * self.fwhm

        if self._n_models == 1:
            w = np.arange(w1, w2, dw)
        else:
            w = list(map(np.arange, w1, w2, dw))

        return np.asarray(w)