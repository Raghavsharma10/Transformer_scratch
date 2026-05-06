def set_data(self, data, offset=None, copy=False):
        """Set texture data

        Parameters
        ----------
        data : ndarray
            Data to be uploaded
        offset: int | tuple of ints
            Offset in texture where to start copying data
        copy: bool
            Since the operation is deferred, data may change before
            data is actually uploaded to GPU memory. Asking explicitly
            for a copy will prevent this behavior.

        Notes
        -----
        This operation implicitely resizes the texture to the shape of
        the data if given offset is None.
        """
        self._set_emulated_shape(data)
        Texture2D.set_data(self, self._normalize_emulated_shape(data),
                           offset, copy)
        self._update_variables()