def _process_version_lines(self):
        """Process version line rules."""
        if len(self._lines_seen["version"]) > 1:
            self._add_error(_("Multiple version lines appeared."))
        elif self._lines_seen["version"][0] != 1:
            self._add_error(_("The version must be on the first line."))