def get_primary_key(self, table):
        """Retrieve the column which is the primary key for a table."""
        for column in self.get_schema(table):
            if len(column) > 3 and 'pri' in column[3].lower():
                return column[0]