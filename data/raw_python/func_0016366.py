def diff_missed_lines(self, filename):
        """
        Return a list of 2-element tuples `(lineno, is_new)` for the given
        file `filename` where `lineno` is a missed line number and `is_new`
        indicates whether the missed line was introduced (True) or removed
        (False).
        """
        line_changed = []
        for line in self.file_source(filename):
            if line.status is not None:
                is_new = not line.status
                line_changed.append((line.number, is_new))
        return line_changed