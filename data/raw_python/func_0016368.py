def file_source_hunks(self, filename):
        """
        Like `CoberturaDiff.file_source`, but returns a list of line hunks of
        the lines that have changed for the given file `filename`. An empty
        list means that the file has no lines that have a change in coverage
        status.
        """
        lines = self.file_source(filename)
        hunks = hunkify_lines(lines)
        return hunks