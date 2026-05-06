def createIndices(self):
        """
        Index columns that are queried. The expression index can
        take a long time.
        """

        sql = '''CREATE INDEX name_index
                 ON Expression (name)'''
        self._cursor.execute(sql)
        self._dbConn.commit()

        sql = '''CREATE INDEX expression_index
                 ON Expression (expression)'''
        self._cursor.execute(sql)
        self._dbConn.commit()