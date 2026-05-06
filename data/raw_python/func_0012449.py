def delete(self, table, where=None, commit=True):
        """
        :type table: string
        :type where: dict
        :type commit: bool
        """
        where_q, _args = self._where_parser(where)

        alias = self._tablename_parser(table)['alias']

        _sql = ''.join(['DELETE ',
                        alias + ' ' if alias else '',
                        'FROM ', self._tablename_parser(table)['formatted_tablename'], where_q, ';'])

        if self.debug:
            return self.cur.mogrify(_sql, _args)

        result = self.cur.execute(_sql, _args)
        if commit:
            self.commit()
        return result