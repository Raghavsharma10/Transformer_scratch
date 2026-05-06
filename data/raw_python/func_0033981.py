def cell(self, rowName, columnName):
        """
        Returns the value of the cell on the given row and column.
        """
        return self.matrix[self.rowIndices[rowName], self.columnIndices[columnName]]