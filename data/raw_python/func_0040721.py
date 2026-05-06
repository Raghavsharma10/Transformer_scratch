def parse(self, lines):
        """Get :class:`base.Result` parameters using regex."""
        pattern = re.compile(r"""^(?P<path>.+?)
                                 :(?P<msg>.+)
                                 :(?P<line_nr>\d+?)
                                 :(?P<col>\d+?)$""", re.VERBOSE)
        return self._parse_by_pattern(lines, pattern)