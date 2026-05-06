def set_data(self, data):
        """ Set the scalar array data

        Parameters
        ----------
        data : ndarray
            A 2D array of scalar values. The isocurve is constructed to show
            all locations in the scalar field equal to ``self.levels``.
        """
        self._data = data

        # if using matplotlib isoline algorithm we have to check for meshgrid
        # and we can setup the tracer object here
        if _HAS_MPL:
            if self._X is None or self._X.T.shape != data.shape:
                self._X, self._Y = np.meshgrid(np.arange(data.shape[0]),
                                               np.arange(data.shape[1]))
            self._iso = cntr.Cntr(self._X, self._Y, self._data.astype(float))

        if self._clim is None:
            self._clim = (data.min(), data.max())

        # sanity check,
        # should we raise an error here, since no isolines can be drawn?
        # for now, _prepare_draw returns False if no isoline can be drawn
        if self._data.min() != self._data.max():
            self._data_is_uniform = False
        else:
            self._data_is_uniform = True

        self._need_recompute = True
        self.update()