def _addFilename(self, filename):
        """
        Add a new file name.

        @param filename: A C{str} file name.
        @raise ValueError: If a file with this name has already been added.
        @return: The C{int} id of the newly added file.
        """
        cur = self._connection.cursor()
        try:
            cur.execute('INSERT INTO files(name) VALUES (?)', (filename,))
        except sqlite3.IntegrityError as e:
            if str(e).find('UNIQUE constraint failed') > -1:
                raise ValueError('Duplicate file name: %r' % filename)
            else:
                raise
        else:
            fileNumber = cur.lastrowid
            self._connection.commit()
            return fileNumber