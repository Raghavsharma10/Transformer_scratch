def _compare_records(self, table, duplicate, options=['r', 'c', 'k', 'sql']):
        """
        Compares similar records and prompts the user to make decisions about keeping, updating, or modifying records in question.

        Parameters
        ----------
        table: str
            The name of the table whose records are being compared.
        duplicate: sequence
            The ids of the potentially duplicate records
        options: list
            The allowed options: 'r' for replace, 'c' for complete, 'k' for keep, 'sql' for raw SQL input.

        """
        # print the old and new records suspected of being duplicates
        verbose = True
        if duplicate[0] == duplicate[1]:  # No need to display if no duplicates were found
            verbose = False

        data = self.query("SELECT * FROM {} WHERE id IN ({})".format(table, ','.join(map(str, duplicate))), \
                          fmt='table', verbose=verbose, use_converters=False)
        columns = data.colnames[1:]
        old, new = [[data[n][k] for k in columns[1:]] for n in [0, 1]]

        # Prompt the user for action
        replace = get_input(
            "\nKeep both records [k]? Or replace [r], complete [c], or keep only [Press *Enter*] record {}? (Type column name to inspect or 'help' for options): ".format(
                    duplicate[0])).lower()
        replace = replace.strip()

        while replace in columns or replace == 'help':
            if replace in columns:
                pprint(np.asarray([[i for idx, i in enumerate(old) if idx in [0, columns.index(replace)]], \
                                   [i for idx, i in enumerate(new) if idx in [0, columns.index(replace)]]]), \
                       names=['id', replace])

            elif replace == 'help':
                _help()

            replace = get_input(
                "\nKeep both records [k]? Or replace [r], complete [c], or keep only [Press *Enter*] record {}? (Type column name to inspect or 'help' for options): ".format(
                        duplicate[0])).lower()

        if replace and replace.split()[0] in options:

            # Replace the entire old record with the new record
            if replace == 'r':
                sure = get_input(
                    'Are you sure you want to replace record {} with record {}? [y/n] : '.format(*duplicate))
                if sure.lower() == 'y':
                    self.modify("DELETE FROM {} WHERE id={}".format(table, duplicate[0]), verbose=False)
                    self.modify("UPDATE {} SET id={} WHERE id={}".format(table, duplicate[0], duplicate[1]),
                                verbose=False)

            # Replace specific columns
            elif replace.startswith('r'):
                replace_cols = replace.split()[1:]
                if all([i in columns for i in replace_cols]):
                    empty_cols, new_vals = zip(
                        *[['{}=?'.format(e), n] for e, n in zip(columns, new) if e in replace_cols])
                    if empty_cols:
                        self.modify("DELETE FROM {} WHERE id={}".format(table, duplicate[1]), verbose=False)
                        self.modify("UPDATE {} SET {} WHERE id={}".format(table, ','.join(empty_cols), duplicate[0]),
                                    tuple(new_vals), verbose=False)
                else:
                    badcols = ','.join([i for i in replace_cols if i not in columns])
                    print("\nInvalid column names for {} table: {}".format(table, badcols))

            # Complete the old record with any missing data provided in the new record, then delete the new record
            elif replace == 'c':
                try:
                    empty_cols, new_vals = zip(
                        *[['{}=?'.format(e), n] for e, o, n in zip(columns[1:], old, new) if n and not o])
                    self.modify("DELETE FROM {} WHERE id={}".format(table, duplicate[1]), verbose=False)
                    self.modify("UPDATE {} SET {} WHERE id={}".format(table, ','.join(empty_cols), duplicate[0]),
                                tuple(new_vals), verbose=False)
                except:
                    pass

            # Keep both records
            elif replace == 'k':
                return 'keep'

            # Execute raw SQL
            elif replace.startswith('sql') and 'sql' in options:
                self.modify(replace[4:], verbose=False)

        # Abort the current database clean up
        elif replace == 'abort':
            return 'abort'

        # Delete the higher id record
        elif not replace:
            self.modify("DELETE FROM {} WHERE id={}".format(table, max(duplicate)), verbose=False)

        # Prompt again
        else:
            print("\nInvalid command: {}\nTry again or type 'help' or 'abort'.\n".format(replace))