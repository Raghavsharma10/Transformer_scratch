def open(self, fname, mode='rb'):
        """
        (Re-)opens a backup file
        """
        self.close()
        self.fp = open(fname, mode)
        self.fname = fname