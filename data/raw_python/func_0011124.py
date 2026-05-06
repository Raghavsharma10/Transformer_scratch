def split(self, sep=None, maxsplit=None, regex=False):
        """Split based on seperator, optionally using a regex

        Capture groups are ignored in regex, the whole pattern is matched
        and used to split the original FmtStr."""
        if maxsplit is not None:
            raise NotImplementedError('no maxsplit yet')
        s = self.s
        if sep is None:
            sep = r'\s+'
        elif not regex:
            sep = re.escape(sep)
        matches = list(re.finditer(sep, s))
        return [self[start:end] for start, end in zip(
            [0] + [m.end() for m in matches],
            [m.start() for m in matches] + [len(s)])]