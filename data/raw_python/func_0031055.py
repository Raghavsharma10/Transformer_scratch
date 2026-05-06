def _getFilename(self, fileNumber):
        """
        Given a file number, get its name (if any).

        @param fileNumber: An C{int} file number.
        @return: A C{str} file name or C{None} if a file with that number
            has not been added.
        """
        cur = self._connection.cursor()
        cur.execute('SELECT name FROM files WHERE id = ?', (fileNumber,))
        row = cur.fetchone()
        if row is None:
            return None
        else:
            return row[0]