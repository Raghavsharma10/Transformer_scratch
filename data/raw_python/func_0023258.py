def set_subdata(self, data, offset=0, copy=False):
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
        """
        data = np.array(data, copy=copy)
        nbytes = data.nbytes

        if offset < 0:
            raise ValueError("Offset must be positive")
        elif (offset + nbytes) > self._nbytes:
            raise ValueError("Data does not fit into buffer")

        # If the whole buffer is to be written, we clear any pending data
        # (because they will be overwritten anyway)
        if nbytes == self._nbytes and offset == 0:
            self._glir.command('SIZE', self._id, nbytes)
        self._glir.command('DATA', self._id, offset, data)