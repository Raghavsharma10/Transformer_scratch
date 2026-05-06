def table(self, table, columns, types, constraints='', pk='', new_table=False):
        """
        Rearrange, add or delete columns from database **table** with desired ordered list of **columns** and corresponding data **types**.

        Parameters
        ----------
        table: sequence
            The name of the table to modify
        columns: list
            A sequence of the columns in the order in which they are to appear in the SQL table
        types: sequence
            A sequence of the types corresponding to each column in the columns list above.
        constraints: sequence (optional)
            A sequence of the constraints for each column, e.g. '', 'UNIQUE', 'NOT NULL', etc.
        pk: string or list
            Name(s) of the primary key(s) if other than ID
        new_table: bool
            Create a new table

        """
        goodtogo = True

        # Make sure there is an integer primary key, unique, not null 'id' column
        # and the appropriate number of elements in each sequence
        if columns[0] != 'id':
            print("Column 1 must be called 'id'")
            goodtogo = False

        if constraints:
            if 'UNIQUE' not in constraints[0].upper() and 'NOT NULL' not in constraints[0].upper():
                print("'id' column constraints must be 'UNIQUE NOT NULL'")
                goodtogo = False
        else:
            constraints = ['UNIQUE NOT NULL'] + ([''] * (len(columns) - 1))

        # Set UNIQUE NOT NULL constraints for the primary keys, except ID which is already has them
        if pk:
            if not isinstance(pk, type(list())):
                pk = list(pk)

            for elem in pk:
                if elem == 'id':
                    continue
                else:
                    ind, = np.where(columns == elem)
                    constraints[ind] = 'UNIQUE NOT NULL'
        else:
            pk = ['id']

        if not len(columns) == len(types) == len(constraints):
            print("Must provide equal length *columns ({}), *types ({}), and *constraints ({}) sequences." \
                  .format(len(columns), len(types), len(constraints)))
            goodtogo = False

        if goodtogo:
            t = self.query("SELECT name FROM sqlite_master", unpack=True, fmt='table')
            tables = t['name'].tolist()

            # If the table exists, modify the columns
            if table in tables and not new_table:

                # Rename the old table and create a new one
                self.list("DROP TABLE IF EXISTS TempOldTable")
                self.list("ALTER TABLE {0} RENAME TO TempOldTable".format(table))
                create_txt = "CREATE TABLE {0} (\n\t{1}".format(table, ', \n\t'.join(
                        ['{} {} {}'.format(c, t, r) for c, t, r in zip(columns, types, constraints)]))
                create_txt += ', \n\tPRIMARY KEY({})\n)'.format(', '.join([elem for elem in pk]))
                # print(create_txt.replace(',', ',\n'))
                self.list(create_txt)

                # Populate the new table and drop the old one
                old_columns = [c for c in self.query("PRAGMA table_info(TempOldTable)", unpack=True)[1] if c in columns]
                self.list("INSERT INTO {0} ({1}) SELECT {1} FROM TempOldTable".format(table, ','.join(old_columns)))

                # Check for and add any foreign key constraints
                t = self.query('PRAGMA foreign_key_list(TempOldTable)', fmt='table')
                if not isinstance(t, type(None)):
                    self.list("DROP TABLE TempOldTable")
                    self.add_foreign_key(table, t['table'].tolist(), t['from'].tolist(), t['to'].tolist())
                else:
                    self.list("DROP TABLE TempOldTable")

            # If the table does not exist and new_table is True, create it
            elif table not in tables and new_table:
                create_txt = "CREATE TABLE {0} (\n\t{1}".format(table, ', \n\t'.join(
                    ['{} {} {}'.format(c, t, r) for c, t, r in zip(columns, types, constraints)]))
                create_txt += ', \n\tPRIMARY KEY({})\n)'.format(', '.join([elem for elem in pk]))
                # print(create_txt.replace(',', ',\n'))
                print(create_txt)
                self.list(create_txt)

            # Otherwise the table to be modified doesn't exist or the new table to add already exists, so do nothing
            else:
                if new_table:
                    print('Table {} already exists. Set *new_table=False to modify.'.format(table.upper()))
                else:
                    print('Table {} does not exist. Could not modify. Set *new_table=True to add a new table.'.format(
                        table.upper()))

        else:
            print('The {} table has not been {}. Please make sure your table columns, \
             types, and constraints are formatted properly.'.format(table.upper(), \
                                                                    'created' if new_table else 'modified'))