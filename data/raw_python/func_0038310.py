def get_schema(self, table, with_headers=False):
        """Retrieve the database schema for a particular table."""
        f = self.fetch('desc ' + wrap(table))
        if not isinstance(f[0], list):
            f = [f]

        # Replace None with ''
        schema = [['' if col is None else col for col in row] for row in f]

        # If with_headers is True, insert headers to first row before returning
        if with_headers:
            schema.insert(0, ['Column', 'Type', 'Null', 'Key', 'Default', 'Extra'])
        return schema