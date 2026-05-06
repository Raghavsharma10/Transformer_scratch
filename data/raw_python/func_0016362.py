def source_lines(self, filename):
        """
        Return a list for source lines of file `filename`.
        """
        with self.filesystem.open(filename) as f:
            return f.readlines()