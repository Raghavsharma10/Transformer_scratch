def _justCreateTable(self, tableClass):
        """
        Execute the table creation DDL for an Item subclass.

        Indexes are *not* created.

        @type tableClass: type
        @param tableClass: an Item subclass
        """
        sqlstr = []
        sqlarg = []

        # needs to be calculated including version
        tableName = self._tableNameFor(tableClass.typeName,
                                       tableClass.schemaVersion)

        sqlstr.append("CREATE TABLE %s (" % tableName)

        # The column is named "oid" instead of "storeID" for backwards
        # compatibility with the implicit oid/rowid column in old Stores.
        sqlarg.append("oid INTEGER PRIMARY KEY")
        for nam, atr in tableClass.getSchema():
            sqlarg.append("\n%s %s" %
                          (atr.getShortColumnName(self), atr.sqltype))

        sqlstr.append(', '.join(sqlarg))
        sqlstr.append(')')
        self.createSQL(''.join(sqlstr))