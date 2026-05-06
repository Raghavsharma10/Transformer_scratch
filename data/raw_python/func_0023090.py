def set_data(self, data=None, **kwargs):
        """Set the line data

        Parameters
        ----------
        data : array-like
            The data.
        **kwargs : dict
            Keywoard arguments to pass to MarkerVisual and LineVisal.
        """
        if data is None:
            pos = None
        else:
            if isinstance(data, tuple):
                pos = np.array(data).T.astype(np.float32)
            else:
                pos = np.atleast_1d(data).astype(np.float32)

            if pos.ndim == 1:
                pos = pos[:, np.newaxis]
            elif pos.ndim > 2:
                raise ValueError('data must have at most two dimensions')

            if pos.size == 0:
                pos = self._line.pos

                # if both args and keywords are zero, then there is no
                # point in calling this function.
                if len(kwargs) == 0:
                    raise TypeError("neither line points nor line properties"
                                    "are provided")
            elif pos.shape[1] == 1:
                x = np.arange(pos.shape[0], dtype=np.float32)[:, np.newaxis]
                pos = np.concatenate((x, pos), axis=1)
            # if args are empty, don't modify position
            elif pos.shape[1] > 3:
                raise TypeError("Too many coordinates given (%s; max is 3)."
                                % pos.shape[1])

        # todo: have both sub-visuals share the same buffers.
        line_kwargs = {}
        for k in self._line_kwargs:
            if k in kwargs:
                k_ = self._kw_trans[k] if k in self._kw_trans else k
                line_kwargs[k] = kwargs.pop(k_)
        if pos is not None or len(line_kwargs) > 0:
            self._line.set_data(pos=pos, **line_kwargs)

        marker_kwargs = {}
        for k in self._marker_kwargs:
            if k in kwargs:
                k_ = self._kw_trans[k] if k in self._kw_trans else k
                marker_kwargs[k_] = kwargs.pop(k)
        if pos is not None or len(marker_kwargs) > 0:
            self._markers.set_data(pos=pos, **marker_kwargs)
        if len(kwargs) > 0:
            raise TypeError("Invalid keyword arguments: %s" % kwargs.keys())