def _set_data(self, data, offset=None, copy=False):
        """Internal method for set_data.
        """
        
        # Copy if needed, check/normalize shape
        data = np.array(data, copy=copy)
        data = self._normalize_shape(data)
        
        # Maybe resize to purge DATA commands?
        if offset is None:
            self._resize(data.shape)
        elif all([i == 0 for i in offset]) and data.shape == self._shape:
            self._resize(data.shape)
        
        # Convert offset to something usable
        offset = offset or tuple([0 for i in range(self._ndim)])
        assert len(offset) == self._ndim
        
        # Check if data fits
        for i in range(len(data.shape)-1):
            if offset[i] + data.shape[i] > self._shape[i]:
                raise ValueError("Data is too large")
        
        # Send GLIR command
        self._glir.command('DATA', self._id, offset, data)