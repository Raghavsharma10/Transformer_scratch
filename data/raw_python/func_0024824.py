def sampleset(self, step=0.01, minimal=False):
        """Return ``x`` array that samples the feature.

        Parameters
        ----------
        step : float
            Distance of first and last points w.r.t. bounding box.

        minimal : bool
            Only return the minimal points needed to define the box;
            i.e., box edges and a point outside on each side.

        """
        w1, w2 = self.bounding_box

        if self._n_models == 1:
            w = self._calc_sampleset(w1, w2, step, minimal)
        else:
            w = list(map(partial(
                self._calc_sampleset, step=step, minimal=minimal), w1, w2))

        return np.asarray(w)