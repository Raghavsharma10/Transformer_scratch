def _find(self, id_):
        """
        Find the filename and offset of a sequence, given its id.

        @param id_: A C{str} sequence id.
        @return: A 2-tuple, containing the C{str} file name and C{int} offset
            within that file of the sequence.
        """
        cur = self._connection.cursor()
        cur.execute(
            'SELECT fileNumber, offset FROM sequences WHERE id = ?', (id_,))
        row = cur.fetchone()
        if row is None:
            return None
        else:
            return self._getFilename(row[0]), row[1]