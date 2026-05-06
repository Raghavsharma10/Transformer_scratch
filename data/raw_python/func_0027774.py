def _indexNameOf(self, tableClass, attrname):
        """
        Return the unqualified (ie, no database name) name of the given
        attribute of the given table.

        @type tableClass: L{MetaItem}
        @param tableClass: The Python class associated with a table in the
            database.

        @param attrname: A sequence of the names of the columns of the
            indicated table which will be included in the named index.

        @return: A C{str} giving the name of the index which will index the
            given attributes of the given table.
        """
        return "axiomidx_%s_v%d_%s" % (tableClass.typeName,
                                       tableClass.schemaVersion,
                                       '_'.join(attrname))