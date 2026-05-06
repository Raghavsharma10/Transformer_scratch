def flush(self):
        """
        Write out any data in the write buffer.  This may do nothing if write
        buffering is not turned on.
        """
        self._write_all(self._wbuffer.getvalue())
        self._wbuffer = StringIO()
        return