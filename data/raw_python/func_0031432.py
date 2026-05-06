def _open(self, filename):
        """
        Open the input file. Set self._fp to point to it. Read the first
        line of parameters.

        @param filename: A C{str} filename containing JSON BLAST records.
        @raise ValueError: if the first line of the file isn't valid JSON,
            if the input file is empty, or if the JSON does not contain an
            'application' key.
        """
        if filename.endswith('.bz2'):
            if six.PY3:
                self._fp = bz2.open(filename, mode='rt', encoding='UTF-8')
            else:
                self._fp = bz2.BZ2File(filename)
        else:
            self._fp = open(filename)

        line = self._fp.readline()
        if not line:
            raise ValueError('JSON file %r was empty.' % self._filename)

        try:
            self.params = loads(line[:-1])
        except ValueError as e:
            raise ValueError(
                'Could not convert first line of %r to JSON (%s). '
                'Line is %r.' % (self._filename, e, line[:-1]))
        else:
            if 'application' not in self.params:
                raise ValueError(
                    '%r appears to be an old JSON file with no BLAST global '
                    'parameters. Please re-run convert-blast-xml-to-json.py '
                    'to convert it to the newest format.' % self._filename)