def __insert(self):
        """Insert rows to table
        """
        if len(self.__buffer) > 0:
            # Insert data
            statement = self.__table.insert()
            if self.__autoincrement:
                statement = statement.returning(
                    getattr(self.__table.c, self.__autoincrement))
                statement = statement.values(self.__buffer)
                res = statement.execute()
                for id, in res:
                    row = self.__buffer.pop(0)
                    yield WrittenRow(row, False, id)
            else:
                statement.execute(self.__buffer)
                for row in self.__buffer:
                    yield WrittenRow(row, False, None)
            # Clean memory
            self.__buffer = []