def getInitialArguments(self):
        """
        Return the constructor arguments required for the JavaScript client class,
        Mantissa.ScrollTable.ScrollTable.

        @return: a 3-tuple of::

          - The unicode attribute ID of my current sort column
          - A list of dictionaries with 'name' and 'type' keys which are
            strings describing the name and type of all the columns in this
            table.
          - A bool indicating whether the sort direction is initially
            ascending.
        """
        ic = IColumn(self.currentSortColumn)
        return [ic.attributeID.decode('ascii'),
                self._getColumnList(),
                self.isAscending]