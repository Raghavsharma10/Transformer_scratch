def rmtree_p(self):
        """ Like :meth:`rmtree`, but does not raise an exception if the
        directory does not exist. """
        try:
            self.rmtree()
        except OSError:
            _, e, _ = sys.exc_info()
            if e.errno != errno.ENOENT:
                raise
        return self