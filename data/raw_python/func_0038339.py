def add_foreign_key(self, table, parent, key_child, key_parent, verbose=True):
        """
        Add foreign key (**key_parent** from **parent**) to **table** column **key_child**

        Parameters
        ----------
        table: string
            The name of the table to modify. This is the child table.
        parent: string or list of strings
            The name of the reference table. This is the parent table.
        key_child: string or list of strings
            Column in **table** to set as foreign key. This is the child key.
        key_parent: string or list of strings
            Column in **parent** that the foreign key refers to. This is the parent key.
        verbose: bool, optional
            Verbose output
        """

        # Temporarily turn off foreign keys
        self.list('PRAGMA foreign_keys=OFF')

        metadata = self.query("PRAGMA table_info({})".format(table), fmt='table')
        columns, types, required, pk = [np.array(metadata[n]) for n in ['name', 'type', 'notnull', 'pk']]

        # Set constraints
        constraints = []
        for elem in required:
            if elem > 0:
                constraints.append('NOT NULL')
            else:
                constraints.append('')

        # Set PRIMARY KEY columns
        ind, = np.where(pk >= 1)
        for i in ind:
            constraints[i] += ' UNIQUE'  # Add UNIQUE constraint to primary keys
        pk_names = columns[ind]

        try:
            # Rename the old table and create a new one
            self.list("DROP TABLE IF EXISTS TempOldTable_foreign")
            self.list("ALTER TABLE {0} RENAME TO TempOldTable_foreign".format(table))

            # Re-create the table specifying the FOREIGN KEY
            sqltxt = "CREATE TABLE {0} (\n\t{1}".format(table, ', \n\t'.join(['{} {} {}'.format(c, t, r)
                                                                      for c, t, r in zip(columns, types, constraints)]))
            sqltxt += ', \n\tPRIMARY KEY({})'.format(', '.join([elem for elem in pk_names]))
            if isinstance(key_child, type(list())):
                for kc, p, kp in zip(key_child, parent, key_parent):
                    sqltxt += ', \n\tFOREIGN KEY ({0}) REFERENCES {1} ({2}) ON UPDATE CASCADE'.format(kc, p, kp)
            else:
                sqltxt += ', \n\tFOREIGN KEY ({0}) REFERENCES {1} ({2}) ON UPDATE CASCADE'.format(key_child, parent, key_parent)
            sqltxt += ' \n)'

            self.list(sqltxt)

            # Populate the new table and drop the old one
            tempdata = self.query("PRAGMA table_info(TempOldTable_foreign)", fmt='table')
            old_columns = [c for c in tempdata['name'] if c in columns]
            self.list("INSERT INTO {0} ({1}) SELECT {1} FROM TempOldTable_foreign".format(table, ','.join(old_columns)))
            self.list("DROP TABLE TempOldTable_foreign")

            if verbose:
                # print('Successfully added foreign key.')
                t = self.query('SELECT name, sql FROM sqlite_master', fmt='table')
                # print(t[t['name'] == table]['sql'][0].replace(',', ',\n'))
                print(t[t['name'] == table]['sql'][0])

        except:
            print('Error attempting to add foreign key.')
            self.list("DROP TABLE IF EXISTS {0}".format(table))
            self.list("ALTER TABLE TempOldTable_foreign RENAME TO {0}".format(table))
            raise sqlite3.IntegrityError('Failed to add foreign key')

        # Reactivate foreign keys
        self.list('PRAGMA foreign_keys=ON')