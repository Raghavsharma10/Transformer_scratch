def init_db(self):
        '''
        Creates all the tables, and indexes needed for the service.

        :return: None
        '''
        with self.conn.transaction() as t:
            t.execute('''
            CREATE TABLE temporal (
                tuid     INTEGER,
                revision CHAR(12) NOT NULL,
                file     TEXT,
                line     INTEGER
            );''')

            t.execute('''
            CREATE TABLE annotations (
                revision       CHAR(12) NOT NULL,
                file           TEXT,
                annotation     TEXT,
                PRIMARY KEY(revision, file)
            );''')

            # Used in frontier updating
            t.execute('''
            CREATE TABLE latestFileMod (
                file           TEXT,
                revision       CHAR(12) NOT NULL,
                PRIMARY KEY(file)
            );''')

            t.execute("CREATE UNIQUE INDEX temporal_rev_file ON temporal(revision, file, line)")
        Log.note("Tables created successfully")