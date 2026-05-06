def compare_schemas(self, db_x, db_y, show=True):
        """
        Compare the structures of two databases.

        Analysis's and compares the column definitions of each table
        in both databases's.  Identifies differences in column names,
        data types and keys.
        """
        # TODO: Improve method
        self._printer("\tComparing database schema's {0} and {1}".format(db_x, db_y))

        # Run compare_dbs_getter to get row counts
        x = self._schema_getter(db_x)
        y = self._schema_getter(db_y)
        x_count = len(x)
        y_count = len(y)

        # Check that database does not have zero tables
        if x_count == 0:
            self._printer('\tThe database {0} has no tables'.format(db_x))
            self._printer('\tDatabase differencing was not run')
            return None
        elif y_count == 0:
            self._printer('\tThe database {0} has no tables'.format(db_y))
            self._printer('\tDatabase differencing was not run')
            return None

        # Print comparisons
        if show:
            uniques_x = diff(x, y, x_only=True)
            if len(uniques_x) > 0:
                self._printer('\nUnique keys from {0} ({1} of {2}):'.format(db_x, len(uniques_x), x_count))
                self._printer('------------------------------')
                # print(uniques)
                for k, v in sorted(uniques_x):
                    self._printer('{0:25} {1}'.format(k, v))
                self._printer('\n')

            uniques_y = diff(x, y, y_only=True)
            if len(uniques_y) > 0:
                self._printer('Unique keys from {0} ({1} of {2}):'.format(db_y, len(uniques_y), y_count))
                self._printer('------------------------------')
                for k, v in sorted(uniques_y):
                    self._printer('{0:25} {1}'.format(k, v))
                self._printer('\n')

            if len(uniques_y) == 0 and len(uniques_y) == 0:
                self._printer("Databases's {0} and {1} are identical:".format(db_x, db_y))
                self._printer('------------------------------')

        return diff(x, y)