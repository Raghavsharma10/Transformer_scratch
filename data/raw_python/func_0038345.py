def info(self):
        """
        Prints out information for the loaded database, namely the available tables and the number of entries for each.
        """
        t = self.query("SELECT * FROM sqlite_master WHERE type='table'", fmt='table')
        all_tables = t['name'].tolist()
        print('\nDatabase path: {} \nSQL path: {}\n'.format(self.dbpath, self.sqlpath))
        print('Database Inventory')
        print('==================')
        for table in ['sources'] + [t for t in all_tables if
                                    t not in ['sources', 'sqlite_sequence']]:
            x = self.query('select count() from {}'.format(table), fmt='array', fetch='one')
            if x is None: continue
            print('{}: {}'.format(table.upper(), x[0]))