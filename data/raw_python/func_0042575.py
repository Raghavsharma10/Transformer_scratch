def recompute_table_revnums(self):
        '''
        Recomputes the revnums for the csetLog table
        by creating a new table, and copying csetLog to
        it. The INTEGER PRIMARY KEY in the temp table auto increments
        as rows are added.

        IMPORTANT: Only call this after acquiring the
                   lock `self.working_locker`.
        :return:
        '''
        with self.conn.transaction() as t:
            t.execute('''
            CREATE TABLE temp (
                revnum         INTEGER PRIMARY KEY,
                revision       CHAR(12) NOT NULL,
                timestamp      INTEGER
            );''')

            t.execute(
                "INSERT INTO temp (revision, timestamp) "
                "SELECT revision, timestamp FROM csetlog ORDER BY revnum ASC"
            )

            t.execute("DROP TABLE csetLog;")
            t.execute("ALTER TABLE temp RENAME TO csetLog;")