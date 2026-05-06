def removedirs_p(self):
        """ Like :meth:`removedirs`, but does not raise an exception if the
        directory is not empty or does not exist. """
        try:
            self.removedirs()
        except OSError:
            _, e, _ = sys.exc_info()
            if e.errno != errno.ENOTEMPTY and e.errno != errno.EEXIST:
                raise
        return self