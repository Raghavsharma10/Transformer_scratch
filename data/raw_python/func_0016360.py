def total_statements(self, filename=None):
        """
        Return the total number of statements for the file
        `filename`. If `filename` is not given, return the total
        number of statements for all files.
        """
        if filename is not None:
            statements = self._get_lines_by_filename(filename)
            return len(statements)

        total = 0
        for filename in self.files():
            statements = self._get_lines_by_filename(filename)
            total += len(statements)

        return total