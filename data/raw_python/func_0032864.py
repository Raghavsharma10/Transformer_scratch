def getTableMetadata(self):
        """
        Retrieve a description of the various properties of this scrolltable.

        @return: A sequence containing 5 elements.  They are, in order, a
        list of the names of the columns present, a mapping of column names
        to two-tuples of their type and a boolean indicating their
        sortability, the total number of rows in the scrolltable, the name
        of the default sort column, and a boolean indicating whether or not
        the current sort order is ascending.
        """
        coltypes = {}
        for (colname, column) in self.columns.iteritems():
            sortable = column.sortAttribute() is not None
            coltype = column.getType()
            if coltype is not None:
                coltype = unicode(coltype, 'ascii')
            coltypes[colname] = (coltype, sortable)

        if self.currentSortColumn:
            csc = unicode(self.currentSortColumn.sortAttribute().attrname, 'ascii')
        else:
            csc = None

        return [self.columnNames, coltypes, self.requestCurrentSize(),
                csc, self.isAscending]