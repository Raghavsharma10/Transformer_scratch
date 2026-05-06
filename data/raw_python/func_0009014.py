def mkdir_p(self, mode=0o777):
        """ Like :meth:`mkdir`, but does not raise an exception if the
        directory already exists. """
        try:
            self.mkdir(mode)
        except OSError:
            _, e, _ = sys.exc_info()
            if e.errno != errno.EEXIST:
                raise
        return self