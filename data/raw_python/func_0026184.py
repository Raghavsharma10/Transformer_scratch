def _decode(self, infile, encoding):
        """
        Decode infile to unicode. Using the specified encoding.

        if is a string, it also needs converting to a list.
        """
        if isinstance(infile, string_types):
            # can't be unicode
            # NOTE: Could raise a ``UnicodeDecodeError``
            return infile.decode(encoding).splitlines(True)
        for i, line in enumerate(infile):
            # NOTE: The isinstance test here handles mixed lists of unicode/string
            # NOTE: But the decode will break on any non-string values
            # NOTE: Or could raise a ``UnicodeDecodeError``
            if PY3K:
                if not isinstance(line, str):
                    infile[i] = line.decode(encoding)
            else:
                if not isinstance(line, unicode):
                    infile[i] = line.decode(encoding)
        return infile