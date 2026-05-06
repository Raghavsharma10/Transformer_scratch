def from_definition(self, table: Table, version: int):
        """Add all columns from the table added in the specified version"""
        self.table(table)
        self.add_columns(*table.columns.get_with_version(version))
        return self