def getSubjectSequence(self, title):
        """
        Obtain information about a subject sequence given its title.

        This information is cached in self._subjectTitleToSubject. It can
        be obtained from either a) an sqlite database (given via the
        sqliteDatabaseFilename argument to __init__), b) the FASTA that was
        originally given to BLAST (via the databaseFilename argument), or
        c) from the BLAST database using blastdbcmd (which can be
        unreliable - occasionally failing to find subjects that are in its
        database).

        @param title: A C{str} sequence title from a BLAST hit. Of the form
            'gi|63148399|gb|DQ011818.1| Description...'.
        @return: An C{AARead} or C{DNARead} instance, depending on the type of
            BLAST database in use.

        """
        if self.params.application in {'blastp', 'blastx'}:
            readClass = AARead
        else:
            readClass = DNARead

        if self._subjectTitleToSubject is None:
            if self._databaseFilename is None:
                if self._sqliteDatabaseFilename is None:
                    # Fall back to blastdbcmd.  ncbidb has to be imported
                    # as below so ncbidb.getSequence can be patched by our
                    # test suite.
                    from dark import ncbidb
                    seq = ncbidb.getSequence(
                        title, self.params.applicationParams['database'])
                    return readClass(seq.description, str(seq.seq))
                else:
                    # An Sqlite3 database is used to look up subjects.
                    self._subjectTitleToSubject = SqliteIndex(
                        self._sqliteDatabaseFilename,
                        fastaDirectory=self._databaseDirectory,
                        readClass=readClass)
            else:
                # Build an in-memory dict to look up subjects. This only
                # works for small databases, obviously.
                titles = {}
                for read in FastaReads(self._databaseFilename,
                                       readClass=readClass):
                    titles[read.id] = read
                self._subjectTitleToSubject = titles

        return self._subjectTitleToSubject[title]