def rows(self):
        """
        Returns a list of dicts.
        """
        rows = []
        for rowName in self.rowNames:
            row = {columnName: self[rowName, columnName] for columnName in self.columnNames}
            row["_"] = rowName
            rows.append(row)
        return rows