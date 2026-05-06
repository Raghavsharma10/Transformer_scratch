def clean_up(self, table, verbose=False):
        """
        Removes exact duplicates, blank records or data without a *source_id* from the specified **table**.
        Then finds possible duplicates and prompts for conflict resolution.

        Parameters
        ----------
        table: str
            The name of the table to remove duplicates, blanks, and data without source attributions.
        verbose: bool
            Print out some diagnostic messages

        """
        # Get the table info and all the records
        metadata = self.query("PRAGMA table_info({})".format(table), fmt='table')
        columns, types, required = [np.array(metadata[n]) for n in ['name', 'type', 'notnull']]
        # records = self.query("SELECT * FROM {}".format(table), fmt='table', use_converters=False)
        ignore = self.query("SELECT * FROM ignore WHERE tablename LIKE ?", (table,))
        duplicate, command = [1], ''

        # Remove records with missing required values
        req_keys = columns[np.where(required)]
        try:
            self.modify("DELETE FROM {} WHERE {}".format(table, ' OR '.join([i + ' IS NULL' for i in req_keys])),
                        verbose=False)
            self.modify(
                "DELETE FROM {} WHERE {}".format(table, ' OR '.join([i + " IN ('null','None','')" for i in req_keys])),
                verbose=False)
        except:
            pass

        # Remove exact duplicates
        self.modify("DELETE FROM {0} WHERE id NOT IN (SELECT min(id) FROM {0} GROUP BY {1})".format(table, ', '.join(
                columns[1:])), verbose=False)

        # Check for records with identical required values but different ids.
        if table.lower() != 'sources': req_keys = columns[np.where(np.logical_and(required, columns != 'id'))]

        # List of old and new pairs to ignore
        if not type(ignore) == np.ndarray: ignore = np.array([])
        new_ignore = []

        while any(duplicate):
            # Pull out duplicates one by one
            if 'source_id' not in columns:  # Check if there is a source_id in the columns
                SQL = "SELECT t1.id, t2.id FROM {0} t1 JOIN {0} t2 ON t1.id=t2.id WHERE ".format(table)
            else:
                SQL = "SELECT t1.id, t2.id FROM {0} t1 JOIN {0} t2 ON t1.source_id=t2.source_id " \
                      "WHERE t1.id!=t2.id AND ".format(table)

            if any(req_keys):
                SQL += ' AND '.join(['t1.{0}=t2.{0}'.format(i) for i in req_keys]) + ' AND '

            if any(ignore):
                SQL += ' AND '.join(
                    ["(t1.id NOT IN ({0}) AND t2.id NOT IN ({0}))".format(','.join(map(str, [id1, id2])))
                        for id1, id2 in zip(ignore['id1'], ignore['id2'])]
                    if any(ignore) else '') + ' AND '

            if any(new_ignore):
                SQL += ' AND '.join(
                    ["(t1.id NOT IN ({0}) AND t2.id NOT IN ({0}))".format(','.join(map(str, ni)))
                     for ni in new_ignore] if new_ignore else '') + ' AND '

            # Clean up empty WHERE at end if it's present (eg, for empty req_keys, ignore, and new_ignore)
            if SQL[-6:] == 'WHERE ':
                SQL = SQL[:-6]

            # Clean up hanging AND if present
            if SQL[-5:] == ' AND ':
                SQL = SQL[:-5]

            if verbose:
                print('\nSearching for duplicates with: {}\n'.format(SQL))

            duplicate = self.query(SQL, fetch='one')

            # Compare potential duplicates and prompt user for action on each
            try:
                # Run record matches through comparison and return the command
                command = self._compare_records(table, duplicate)

                # Add acceptable duplicates to ignore list or abort
                if command == 'keep':
                    new_ignore.append([duplicate[0], duplicate[1]])
                    self.list("INSERT INTO ignore VALUES(?,?,?,?)", (None, duplicate[0], duplicate[1], table.lower()))
                elif command == 'undo':
                    pass  # TODO: Add this functionality!
                elif command == 'abort':
                    break
                else:
                    pass
            except:
                break

        # Finish or abort table clean up
        if command == 'abort':
            print('\nAborted clean up of {} table.'.format(table.upper()))
            return 'abort'
        else:
            print('\nFinished clean up on {} table.'.format(table.upper()))