def tables(self):
        """
        Get a listing of all tables
          - if schema specified on connect, return unqualifed table names in
            that schema
          - in no schema specified on connect, return all tables, with schema
            prefixes
        """
        if self.schema:
            return self.tables_in_schema(self.schema)
        else:
            tables = []
            for schema in self.schemas:
                tables = tables + [
                    schema + "." + t for t in self.tables_in_schema(schema)
                ]
            return tables