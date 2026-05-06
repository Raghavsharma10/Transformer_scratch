def check(self, final_line_count):
        """Check the status of all provided data and update the suite."""
        if self._lines_seen["version"]:
            self._process_version_lines()
        self._process_plan_lines(final_line_count)