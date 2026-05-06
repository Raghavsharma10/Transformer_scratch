def set_data(self, image):
        """Set the data

        Parameters
        ----------
        image : array-like
            The image data.
        """
        data = np.asarray(image)
        if self._data is None or self._data.shape != data.shape:
            self._need_vertex_update = True
        self._data = data
        self._need_texture_upload = True