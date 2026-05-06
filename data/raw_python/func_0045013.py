def next(self):
        """
        Yields the next row from the source files.
        """
        for self._filename in self._filenames:
            self._open()
            for row in self._csv_reader:
                self._row_number += 1
                if self._fields:
                    yield dict(zip_longest(self._fields, row, fillvalue=''))
                else:
                    yield row
            self._close()
            self._row_number = -1

        self._filename = None
        raise StopIteration