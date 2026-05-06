def save(self, output=None):
        """
        Save the database to a file. If ``output`` is not given, the ``dbfile``
        given in the constructor is used.
        """
        if output is None:
            if self.dbfile is None:
                return
            output = self.dbfile

        with open(output, 'w') as f:
            f.write(self._dump())