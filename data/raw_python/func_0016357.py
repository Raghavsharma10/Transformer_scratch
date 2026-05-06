def file_source(self, filename):
        """
        Return a list of namedtuple `Line` for each line of code found in the
        source file with the given `filename`.
        """
        lines = []
        try:
            with self.filesystem.open(filename) as f:
                line_statuses = dict(self.line_statuses(filename))
                for lineno, source in enumerate(f, start=1):
                    line_status = line_statuses.get(lineno)
                    line = Line(lineno, source, line_status, None)
                    lines.append(line)

        except self.filesystem.FileNotFound as file_not_found:
            lines.append(
                Line(0, '%s not found' % file_not_found.path, None, None)
            )

        return lines