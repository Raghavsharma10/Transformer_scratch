def _getFileNumber(self, filename):
        """
        Given a file name, get its file number (if any).

        @param filename: A C{str} file name.
        @return: An C{int} file number or C{None} if no file with that name
            has been added.
        """
        cur = self._connection.cursor()
        cur.execute('SELECT id FROM files WHERE name = ?', (filename,))
        row = cur.fetchone()
        if row is None:
            return None
        else:
            return row[0]