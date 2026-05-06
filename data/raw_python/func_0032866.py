def _getColumnList(self):
        """
        Get a list of serializable objects that describe the interesting
        columns on our item type.  Columns which report having no type will be
        treated as having the type I{text}.

        @rtype: C{list} of C{dict}
        """
        columnList = []
        for columnName in self.columnNames:
            column = self.columns[columnName]
            type = column.getType()
            if type is None:
                type = 'text'
            columnList.append(
                {u'name': columnName,
                 u'type': type.decode('ascii')})
        return columnList