def _getReader(self, filename, scoreClass):
        """
        Obtain a JSON record reader for DIAMOND records.

        @param filename: The C{str} file name holding the JSON.
        @param scoreClass: A class to hold and compare scores (see scores.py).
        """
        if filename.endswith('.json') or filename.endswith('.json.bz2'):
            return JSONRecordsReader(filename, scoreClass)
        else:
            raise ValueError(
                'Unknown DIAMOND record file suffix for file %r.' % filename)