def insert(self, table, value, ignore=False, commit=True):
        """
        Insert a dict into db.
        :type table: string
        :type value: dict
        :type ignore: bool
        :type commit: bool
        :return: int. The row id of the insert.
        """
        value_q, _args = self._value_parser(value, columnname=False)
        _sql = ''.join(['INSERT', ' IGNORE' if ignore else '', ' INTO ', self._backtick(table),
                        ' (', self._backtick_columns(value), ') VALUES (', value_q, ');'])

        if self.debug:
            return self.cur.mogrify(_sql, _args)

        self.cur.execute(_sql, _args)
        if commit:
            self.conn.commit()
        return self.cur.lastrowid