def _computeFromClause(self, tables):
        """
        Generate the SQL string which follows the "FROM" string and before the
        "WHERE" string in the final SQL statement.
        """
        tableAliases = []
        self.fromClauseParts = []
        for table in tables:
            # The indirect calls to store.getTableName() will create the tables
            # if needed. (XXX That's bad, actually.   They should get created
            # some other way if necessary.  -exarkun)
            tableName = table.getTableName(self.store)
            tableAlias = table.getTableAlias(self.store, tuple(tableAliases))
            if tableAlias is None:
                self.fromClauseParts.append(tableName)
            else:
                tableAliases.append(tableAlias)
                self.fromClauseParts.append('%s AS %s' % (tableName,
                                                          tableAlias))

        self.sortClauseParts = []
        for attr, direction in self.sort.orderColumns():
            assert direction in ('ASC', 'DESC'), "%r not in ASC,DESC" % (direction,)
            if attr.type not in tables:
                raise ValueError(
                    "Ordering references type excluded from comparison")
            self.sortClauseParts.append(
                '%s %s' % (attr.getColumnName(self.store), direction))