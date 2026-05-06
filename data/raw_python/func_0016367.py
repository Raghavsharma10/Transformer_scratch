def file_source(self, filename):
        """
        Return a list of namedtuple `Line` for each line of code found in the
        given file `filename`.

        """
        if self.cobertura1.has_file(filename) and \
                self.cobertura1.filesystem.has_file(filename):
            lines1 = self.cobertura1.source_lines(filename)
            line_statuses1 = dict(self.cobertura1.line_statuses(
                filename))
        else:
            lines1 = []
            line_statuses1 = {}

        lines2 = self.cobertura2.source_lines(filename)
        line_statuses2 = dict(self.cobertura2.line_statuses(filename))

        # Build a dict of lineno2 -> lineno1
        lineno_map = reconcile_lines(lines2, lines1)

        lines = []
        for lineno, source in enumerate(lines2, start=1):
            status = None
            reason = None
            if lineno not in lineno_map:
                # line was added or removed, just use whatever coverage status
                # is available as there is nothing to compare against.
                status = line_statuses2.get(lineno)
                reason = 'line-edit'
            else:
                other_lineno = lineno_map[lineno]
                line_status1 = line_statuses1.get(other_lineno)
                line_status2 = line_statuses2.get(lineno)
                if line_status1 is line_status2:
                    status = None  # unchanged
                    reason = None
                elif line_status1 is True and line_status2 is False:
                    status = False  # decreased
                    reason = 'cov-down'
                elif line_status1 is False and line_status2 is True:
                    status = True  # increased
                    reason = 'cov-up'

            line = Line(lineno, source, status, reason)
            lines.append(line)

        return lines