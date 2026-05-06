def set_data(self, data, copy=False, **kwargs):
        """ Set data (deferred operation)

        Parameters
        ----------
        data : ndarray
            Data to be uploaded
        copy: bool
            Since the operation is deferred, data may change before
            data is actually uploaded to GPU memory.
            Asking explicitly for a copy will prevent this behavior.
        **kwargs : dict
            Additional arguments.
        """
        data = self._prepare_data(data, **kwargs)
        self._dtype = data.dtype
        self._stride = data.strides[-1]
        self._itemsize = self._dtype.itemsize
        Buffer.set_data(self, data=data, copy=copy)