def set_data(self, data, copy=False):
        """ Set data in the buffer (deferred operation).

        This completely resets the size and contents of the buffer.

        Parameters
        ----------
        data : ndarray
            Data to be uploaded
        copy: bool
            Since the operation is deferred, data may change before
            data is actually uploaded to GPU memory.
            Asking explicitly for a copy will prevent this behavior.
        """
        data = np.array(data, copy=copy)
        nbytes = data.nbytes

        if nbytes != self._nbytes:
            self.resize_bytes(nbytes)
        else:
            # Use SIZE to discard any previous data setting
            self._glir.command('SIZE', self._id, nbytes)
        
        if nbytes:  # Only set data if there *is* data
            self._glir.command('DATA', self._id, 0, data)