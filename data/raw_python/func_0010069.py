def save(self, filename=None):
        """
        Save an arff structure to a file.
        """
        filename = filename or self._filename
        o = open(filename, 'w')
        o.write(self.write())
        o.close()