def parse(self, fh):
        """Generate tap.line.Line objects, given a file-like object `fh`.

        `fh` may be any object that implements both the iterator and
        context management protocol (i.e. it can be used in both a
        "with" statement and a "for...in" statement.)

        Trailing whitespace and newline characters will be automatically
        stripped from the input lines.
        """
        with fh:
            try:
                first_line = next(fh)
            except StopIteration:
                return
            first_parsed = self.parse_line(first_line.rstrip())
            fh_new = itertools.chain([first_line], fh)
            if first_parsed.category == "version" and first_parsed.version >= 13:
                if ENABLE_VERSION_13:
                    fh_new = peekable(itertools.chain([first_line], fh))
                    self._try_peeking = True
                else:  # pragma no cover
                    print(
                        """
WARNING: Optional imports not found, TAP 13 output will be
    ignored. To parse yaml, see requirements in docs:
    https://tappy.readthedocs.io/en/latest/consumers.html#tap-version-13"""
                    )

            for line in fh_new:
                yield self.parse_line(line.rstrip(), fh_new)