def missed_lines(self, filename):
        """
        Return a list of extrapolated uncovered line numbers for the
        file `filename` according to `Cobertura.line_statuses`.
        """
        statuses = self.line_statuses(filename)
        statuses = extrapolate_coverage(statuses)
        return [lno for lno, status in statuses if status is False]