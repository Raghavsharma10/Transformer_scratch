def open(self, *args, **kwargs):
        """ Open this file and return a corresponding :class:`file` object.

        Keyword arguments work as in :func:`io.open`.  If the file cannot be
        opened, an :class:`~exceptions.OSError` is raised.
        """
        with io_error_compat():
            return io.open(self, *args, **kwargs)