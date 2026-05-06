def snapshot(self, name_db='export.db', version=1.0):
        """
        Function to generate a snapshot of the database by version number.

        Parameters
        ----------
        name_db: string
            Name of the new database (Default: export.db)
        version: float
            Version number to export (Default: 1.0)
        """

        # Check if file exists
        if os.path.isfile(name_db):
            import datetime
            date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
            print("Renaming existing file {} to {}".format(name_db, name_db.replace('.db', date + '.db')))
            os.system("mv {} {}".format(name_db, name_db.replace('.db', date + '.db')))

        # Create a new database from existing database schema
        t, = self.query("select sql from sqlite_master where type = 'table'", unpack=True)
        schema = ';\n'.join(t) + ';'
        os.system("sqlite3 {} '{}'".format(name_db, schema))

        # Attach database to newly created database
        db = Database(name_db)
        db.list('PRAGMA foreign_keys=OFF')  # Temporarily deactivate foreign keys for the snapshot
        db.list("ATTACH DATABASE '{}' AS orig".format(self.dbpath))

        # For each table in database, insert records if they match version number
        t = db.query("SELECT * FROM sqlite_master WHERE type='table'", fmt='table')
        all_tables = t['name'].tolist()
        for table in [t for t in all_tables if t not in ['sqlite_sequence', 'changelog']]:
            # Check if this is table with version column
            metadata = db.query("PRAGMA table_info({})".format(table), fmt='table')
            columns, types, required, pk = [np.array(metadata[n]) for n in ['name', 'type', 'notnull', 'pk']]
            print(table, columns)
            if 'version' not in columns:
                db.modify("INSERT INTO {0} SELECT * FROM orig.{0}".format(table))
            else:
                db.modify("INSERT INTO {0} SELECT * FROM orig.{0} WHERE orig.{0}.version<={1}".format(table, version))

        # Detach original database
        db.list('DETACH DATABASE orig')
        db.list('PRAGMA foreign_keys=ON')
        db.close(silent=True)