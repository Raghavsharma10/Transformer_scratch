def update(self, fname):
        """
        Adds a handler to save to a file. Includes debug stuff.
        """
        ltfh = FileHandler(fname)
        self._log.addHandler(ltfh)