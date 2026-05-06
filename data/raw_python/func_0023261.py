def set_subdata(self, data, offset=0, copy=False, **kwargs):
        """ Set a sub-region of the buffer (deferred operation).

        Parameters
        ----------

        data : ndarray
            Data to be uploaded
        offset: int
            Offset in buffer where to start copying data (in bytes)
        copy: bool
            Since the operation is deferred, data may change before
            data is actually uploaded to GPU memory.
            Asking explicitly for a copy will prevent this behavior.
        **kwargs : dict
            Additional keyword arguments.
        """
        data = self._prepare_data(data, **kwargs)
        offset = offset * self.itemsize
        Buffer.set_subdata(self, data=data, offset=offset, copy=copy)