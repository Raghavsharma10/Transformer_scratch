def merge(self, conflicted, tables=[], diff_only=True):
        """
        Merges specific **tables** or all tables of **conflicted** database into the master database.

        Parameters
        ----------
        conflicted: str
            The path of the SQL database to be merged into the master.
        tables: list (optional)
            The list of tables to merge. If None, all tables are merged.
        diff_only: bool
                If True, only prints the differences of each table and doesn't actually merge anything.

        """
        if os.path.isfile(conflicted):
            # Load and attach master and conflicted databases
            con, master, reassign = Database(conflicted), self.list("PRAGMA database_list").fetchall()[0][2], {}
            con.modify("ATTACH DATABASE '{}' AS m".format(master), verbose=False)
            self.modify("ATTACH DATABASE '{}' AS c".format(conflicted), verbose=False)
            con.modify("ATTACH DATABASE '{}' AS c".format(conflicted), verbose=False)
            self.modify("ATTACH DATABASE '{}' AS m".format(master), verbose=False)

            # Drop any backup tables from failed merges
            for table in tables: self.modify("DROP TABLE IF EXISTS Backup_{0}".format(table), verbose=False)

            # Gather user data to add to CHANGELOG table
            import socket, datetime
            if not diff_only: user = get_input('Please enter your name : ')
            machine_name = socket.gethostname()
            date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            modified_tables = []

            # Merge table by table, starting with SOURCES
            if not isinstance(tables, type(list())):
                tables = [tables]

            tables = tables or ['sources'] + [t for t in zip(*self.list(
                "SELECT * FROM sqlite_master WHERE name NOT LIKE '%Backup%' AND name!='sqlite_sequence' AND type='table'{}".format(
                    " AND name IN ({})".format("'" + "','".join(tables) + "'") if tables else '')))[1] if
                                              t != 'sources']
            for table in tables:
                # Get column names and data types from master table and column names from conflicted table
                metadata = self.query("PRAGMA table_info({})".format(table), fmt='table')
                columns, types, constraints = [np.array(metadata[n]) for n in ['name', 'type', 'notnull']]
                # columns, types, constraints = self.query("PRAGMA table_info({})".format(table), unpack=True)[1:4]
                conflicted_cols = con.query("PRAGMA table_info({})".format(table), unpack=True)[1]

                if any([i not in columns for i in conflicted_cols]):
                    # Abort table merge if conflicted has new columns not present in master. New columns must be added to the master database first via db.edit_columns().
                    print(
                    "\nMerge of {0} table aborted since conflicted copy has columns {1} not present in master.\nAdd new columns to master with astrodb.table() method and try again.\n".format(
                        table.upper(), [i for i in conflicted_cols if i not in columns]))

                else:
                    # Add new columns from master table to conflicted table if necessary
                    if any([i not in conflicted_cols for i in columns]):
                        con.modify("DROP TABLE IF EXISTS Conflicted_{0}".format(table))
                        con.modify("ALTER TABLE {0} RENAME TO Conflicted_{0}".format(table))
                        # TODO: Update to allow multiple primary and foreign keys
                        con.modify("CREATE TABLE {0} ({1})".format(table, ', '.join( \
                                ['{} {} {}{}'.format(c, t, r, ' UNIQUE PRIMARY KEY' if c == 'id' else '') \
                                 for c, t, r in zip(columns, types, constraints * ['NOT NULL'])])))
                        con.modify("INSERT INTO {0} ({1}) SELECT {1} FROM Conflicted_{0}".format(table, ','.join(
                            conflicted_cols)))
                        con.modify("DROP TABLE Conflicted_{0}".format(table))

                    # Pull unique records from conflicted table
                    data = map(list, con.list(
                        "SELECT * FROM (SELECT 1 AS db, {0} FROM m.{2} UNION ALL SELECT 2 AS db, {0} FROM c.{2}) GROUP BY {1} HAVING COUNT(*)=1 AND db=2".format(
                            ','.join(columns), ','.join(columns[1:]), table)).fetchall())

                    if data:

                        # Just print the table differences
                        if diff_only:
                            pprint(zip(*data)[1:], names=columns, title='New {} records'.format(table.upper()))

                        # Add new records to the master and then clean up tables
                        else:
                            # Make temporary table copy so changes can be undone at any time
                            self.modify("DROP TABLE IF EXISTS Backup_{0}".format(table), verbose=False)
                            self.modify("ALTER TABLE {0} RENAME TO Backup_{0}".format(table), verbose=False)
                            self.modify("CREATE TABLE {0} ({1})".format(table, ', '.join( \
                                    ['{} {} {}{}'.format(c, t, r, ' UNIQUE PRIMARY KEY' if c == 'id' else '') \
                                     for c, t, r in zip(columns, types, constraints * ['NOT NULL'])])), verbose=False)
                            self.modify(
                                "INSERT INTO {0} ({1}) SELECT {1} FROM Backup_{0}".format(table, ','.join(columns)),
                                verbose=False)

                            # Create a dictionary of any reassigned ids from merged SOURCES tables and replace applicable source_ids in other tables.
                            print("\nMerging {} tables.\n".format(table.upper()))
                            try:
                                count = self.query("SELECT MAX(id) FROM {}".format(table), fetch='one')[0] + 1
                            except TypeError:
                                count = 1
                            for n, i in enumerate([d[1:] for d in data]):
                                if table == 'sources':
                                    reassign[i[0]] = count
                                elif 'source_id' in columns and i[1] in reassign.keys():
                                    i[1] = reassign[i[1]]
                                else:
                                    pass
                                i[0] = count
                                data[n] = i
                                count += 1

                            # Insert unique records into master
                            for d in data: self.modify(
                                "INSERT INTO {} VALUES({})".format(table, ','.join(['?' for c in columns])), d,
                                verbose=False)
                            pprint(zip(*data), names=columns,
                                   title="{} records added to {} table at '{}':".format(len(data), table, master))

                            # Run clean_up on the table to check for conflicts
                            abort = self.clean_up(table)

                            # Undo all changes to table if merge is aborted. Otherwise, push table changes to master.
                            if abort:
                                self.modify("DROP TABLE {0}".format(table), verbose=False)
                                self.modify("ALTER TABLE Backup_{0} RENAME TO {0}".format(table), verbose=False)
                            else:
                                self.modify("DROP TABLE Backup_{0}".format(table), verbose=False)
                                modified_tables.append(table.upper())

                    else:
                        print("\n{} tables identical.".format(table.upper()))

            # Add data to CHANGELOG table
            if not diff_only:
                user_description = get_input('\nPlease describe the changes made in this merge: ')
                self.list("INSERT INTO changelog VALUES(?, ?, ?, ?, ?, ?, ?)", \
                          (None, date, str(user), machine_name, ', '.join(modified_tables), user_description,
                           os.path.basename(conflicted)))

            # Finish up and detach
            if diff_only:
                print("\nDiff complete. No changes made to either database. Set `diff_only=False' to apply merge.")
            else:
                print("\nMerge complete!")

            con.modify("DETACH DATABASE c", verbose=False)
            self.modify("DETACH DATABASE c", verbose=False)
            con.modify("DETACH DATABASE m", verbose=False)
            self.modify("DETACH DATABASE m", verbose=False)
        else:
            print("File '{}' not found!".format(conflicted))