def insertmany(self, table, columns, value, ignore=False, commit=True):
        """
        Insert multiple records within one query.
        :type table: string
        :type columns: list
        :type value: list|tuple
        :param value: Doesn't support MySQL functions
        :param value: Example: [(value1_column1, value1_column2,), ]
        :type ignore: bool
        :type commit: bool
        :return: int. The row id of the LAST insert only.
        """
        if not isinstance(value, (list, tuple)):
            raise TypeError('Input value should be a list or tuple')

        # Cannot add semicolon here, otherwise it will not pass the Cursor.executemany validation
        _sql = ''.join(['INSERT', ' IGNORE' if ignore else '', ' INTO ', self._backtick(table),
                        ' (', self._backtick_columns(columns), ') VALUES (', ', '.join(['%s'] * len(columns)), ')'])
        _args = tuple(value)

        # For insertmany, the base queries for executemany and printing are different
        _sql_full = ''.join(['INSERT', ' IGNORE' if ignore else '', ' INTO ', self._backtick(table),
                             ' (', self._backtick_columns(columns), ') VALUES ',
                             ', '.join([''.join(['(', ', '.join(['%s'] * len(columns)), ')'])] * len(_args)),
                             ';'])

        _args_flattened = [item for sublist in _args for item in sublist]

        if self.debug:
            return self.cur.mogrify(_sql_full, _args_flattened)

        self.cur.executemany(_sql, _args)
        if commit:
            self.conn.commit()
        return self.cur.lastrowid