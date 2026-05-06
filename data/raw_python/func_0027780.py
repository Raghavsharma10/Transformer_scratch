def _createIndexesFor(self, tableClass, extantIndexes):
        """
        Create any indexes which don't exist and are required by the schema
        defined by C{tableClass}.

        @param tableClass: A L{MetaItem} instance which may define a schema
            which includes indexes.

        @param extantIndexes: A container (anything which can be the right-hand
            argument to the C{in} operator) which contains the unqualified
            names of all indexes which already exist in the underlying database
            and do not need to be created.
        """
        try:
            indexes = _requiredTableIndexes[tableClass]
        except KeyError:
            indexes = set()
            for nam, atr in tableClass.getSchema():
                if atr.indexed:
                    indexes.add(((atr.getShortColumnName(self),), (atr.attrname,)))
                for compound in atr.compoundIndexes:
                    indexes.add((tuple(inatr.getShortColumnName(self) for inatr in compound),
                                 tuple(inatr.attrname for inatr in compound)))
            _requiredTableIndexes[tableClass] = indexes

        # _ZOMFG_ SQL is such a piece of _shit_: you can't fully qualify the
        # table name in CREATE INDEX statements because the _INDEX_ is fully
        # qualified!

        indexColumnPrefix = '.'.join(self.getTableName(tableClass).split(".")[1:])

        for (indexColumns, indexAttrs) in indexes:
            nameOfIndex = self._indexNameOf(tableClass, indexAttrs)
            if nameOfIndex in extantIndexes:
                continue
            csql = 'CREATE INDEX %s.%s ON %s(%s)' % (
                self.databaseName, nameOfIndex, indexColumnPrefix,
                ', '.join(indexColumns))
            self.createSQL(csql)