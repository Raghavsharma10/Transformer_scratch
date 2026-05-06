def convert(self, line=None, is_end=True):
        """Read the line content and return the converted value

        :param line: the line to feed to converter
        :param is_end: if set to True, will raise an error if
        the line has something remaining.
        """
        if line is not None:
            self.line = line
        if not self.line:
            raise TomlDecodeError(self.parser.lineno,
                                  'EOF is hit!')
        token = None
        self.line = self.line.lstrip()
        for key, pattern in self.patterns:
            m = pattern.match(self.line)
            if m:
                self.line = self.line[m.end():]
                handler = getattr(self, 'convert_%s' % key)
                token = handler(m)
                break
        else:
            raise TomlDecodeError(self.parser.lineno,
                                  'Parsing error: %r' % self.line)
        if is_end and not BLANK_RE.match(self.line):
            raise TomlDecodeError(self.parser.lineno,
                                  'Something is remained: %r' % self.line)
        return token