def update(self, table, value, where, join=None, commit=True):
        """
        :type table: string
        :type value: dict
        :type where: dict
        :type join: dict
        :type commit: bool
        """

        value_q, _value_args = self._value_parser(value, columnname=True)

        where_q, _where_args = self._where_parser(where)

        _sql = ''.join(['UPDATE ', self._tablename_parser(table)['formatted_tablename'],
                        self._join_parser(join),
                        ' SET ', value_q, where_q, ';'])
        _args = _value_args + _where_args

        if self.debug:
            return self.cur.mogrify(_sql, _args)

        result = self.cur.execute(_sql, _args)
        if commit:
            self.commit()
        return result