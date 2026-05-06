def _get_tap_file_path(self, test_case):
        """Get the TAP output file path for the test case."""
        sanitized_test_case = test_case.translate(self._sanitized_table)
        tap_file = sanitized_test_case + ".tap"
        if self.outdir:
            return os.path.join(self.outdir, tap_file)
        return tap_file