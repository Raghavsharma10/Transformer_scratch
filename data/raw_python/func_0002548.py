def parse_line(self, text, fh=None):
        """Parse a line into whatever TAP category it belongs."""
        match = self.ok.match(text)
        if match:
            return self._parse_result(True, match, fh)

        match = self.not_ok.match(text)
        if match:
            return self._parse_result(False, match, fh)

        if self.diagnostic.match(text):
            return Diagnostic(text)

        match = self.plan.match(text)
        if match:
            return self._parse_plan(match)

        match = self.bail.match(text)
        if match:
            return Bail(match.group("reason"))

        match = self.version.match(text)
        if match:
            return self._parse_version(match)

        return Unknown()