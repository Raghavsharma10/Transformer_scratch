def getSubjectSequence(self, title):
        """
        Obtain information about a subject sequence given its title.

        @param title: A C{str} sequence title from a DIAMOND hit.
        @raise KeyError: If the C{title} is not present in the DIAMOND
            database.
        @return: An C{AAReadWithX} instance.
        """
        if self._subjectTitleToSubject is None:
            if self._databaseFilename is None:
                # An Sqlite3 database is used to look up subjects.
                self._subjectTitleToSubject = SqliteIndex(
                    self._sqliteDatabaseFilename,
                    fastaDirectory=self._databaseDirectory,
                    readClass=AAReadWithX)
            else:
                # Build a dict to look up subjects.
                titles = {}
                for read in FastaReads(self._databaseFilename,
                                       readClass=AAReadWithX):
                    titles[read.id] = read
                self._subjectTitleToSubject = titles

        return self._subjectTitleToSubject[title]