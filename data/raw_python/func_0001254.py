def drop_schema(self, schema, cascade=False):
        """Drop specified schema
        """
        if schema in self.schemas:
            sql = "DROP SCHEMA " + schema
            if cascade:
                sql = sql + " CASCADE"
            self.execute(sql)