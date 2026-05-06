def next(self):
        """
        Gets next entry as a dictionary.

        Returns:
            object - Object key/value pair representing a row. 
            {key1: value1, key2: value2, ...}

        """
        try:
            entry = {}
            row = self._csv_reader.next()
            for i in range(0, len(row)):
                entry[self._headers[i]] = row[i]

            return entry
        except Exception as e:
            # close our file when we're done reading.
            self._file.close()
            raise e