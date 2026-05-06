def rename(self, old_table, new_table):
        """
        Rename a table.

        You must have ALTER and DROP privileges for the original table,
        and CREATE and INSERT privileges for the new table.
        """
        try:
            command = 'RENAME TABLE {0} TO {1}'.format(wrap(old_table), wrap(new_table))
        except:
            command = 'ALTER TABLE {0} RENAME {1}'.format(wrap(old_table), wrap(new_table))
        self.execute(command)
        self._printer('Renamed {0} to {1}'.format(wrap(old_table), wrap(new_table)))
        return old_table, new_table