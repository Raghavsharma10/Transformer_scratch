def total_misses(self, filename=None):
        """
        Return the total number of uncovered statements for the file
        `filename`. If `filename` is not given, return the total
        number of uncovered statements for all files.
        """
        if filename is not None:
            return len(self.missed_statements(filename))

        total = 0
        for filename in self.files():
            total += len(self.missed_statements(filename))

        return total