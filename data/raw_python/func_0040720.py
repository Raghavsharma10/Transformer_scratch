def parse(self, lines):
        """Get :class:`base.Result` parameters using regex.

        There are 2 lines for each pydocstyle result:
            1. Filename and line number;
            2. Message for the problem found.
        """
        patterns = [re.compile(r'^(.+?):(\d+)'),
                    re.compile(r'^\s+(.+)$')]
        for i, line in enumerate(lines):
            if i % 2 == 0:
                path, line_nr = patterns[0].match(line).groups()
            else:
                msg = patterns[1].match(line).group(1)
                yield LinterOutput(self.name, path, msg, line_nr)