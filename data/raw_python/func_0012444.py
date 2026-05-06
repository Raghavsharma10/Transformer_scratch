def get(self, table, column, join=None, where=None, insert=False, ifnone=None):
        """
        A simplified method of select, for getting the first result in one column only. A common case of using this
        method is getting id.
        :type table: string
        :type column: str
        :type join: dict
        :type where: dict
        :type insert: bool
        :param insert: If insert==True, insert the input condition if there's no result and return the id of new row.
        :type ifnone: string
        :param ifnone: When ifnone is a non-empty string, raise an error if query returns empty result. insert parameter
                       would not work in this mode.
        """
        select_result = self.select(table=table, columns=[column], join=join, where=where, limit=1)

        if self.debug:
            return select_result

        result = select_result[0] if select_result else None

        if result:
            return result[0 if self.cursorclass is pymysql.cursors.Cursor else column]

        if ifnone:
            raise ValueError(ifnone)

        if insert:
            if any([isinstance(d, dict) for d in where.values()]):
                raise ValueError("The where parameter in get() doesn't support nested condition with insert==True.")
            return self.insert(table=table, value=where)

        return None