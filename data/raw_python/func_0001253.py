def create_schema(self, schema):
        """Create specified schema if it does not already exist
        """
        if schema not in self.schemas:
            sql = "CREATE SCHEMA " + schema
            self.execute(sql)