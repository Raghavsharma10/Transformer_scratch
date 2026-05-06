def remove_p(self):
        """ Like :meth:`remove`, but does not raise an exception if the
        file does not exist. """
        try:
            self.unlink()
        except OSError:
            _, e, _ = sys.exc_info()
            if e.errno != errno.ENOENT:
                raise
        return self