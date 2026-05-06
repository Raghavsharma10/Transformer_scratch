def close(self):
        '''Close the underlying file.

        Sets data attribute .closed to True. A closed file cannot be used for
        further I/O operations. close() may be called more than once without
        error. Some kinds of file objects (for example, opened by popen())
        may return an exit status upon closing.'''

        if self._mode in _allowed_write and self._valid is None:
            self._finalize_write()
        result = self._fp.close()
        self._closed = True

        return result