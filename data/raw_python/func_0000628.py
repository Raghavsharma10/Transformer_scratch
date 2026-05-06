def __update(self, row):
        """Update rows in table
        """
        expr = self.__table.update().values(row)
        for key in self.__update_keys:
            expr = expr.where(getattr(self.__table.c, key) == row[key])
        if self.__autoincrement:
            expr = expr.returning(getattr(self.__table.c, self.__autoincrement))
        res = expr.execute()
        if res.rowcount > 0:
            if self.__autoincrement:
                first = next(iter(res))
                last_row_id = first[0]
                return last_row_id
            return 0
        return None