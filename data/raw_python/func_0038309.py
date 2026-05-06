def get_schema_dict(self, table):
        """
        Retrieve the database schema in key, value pairs for easier
        references and comparisons.
        """
        # Retrieve schema in list form
        schema = self.get_schema(table, with_headers=True)

        # Pop headers from first item in list
        headers = schema.pop(0)

        # Create dictionary by zipping headers with each row
        return {values[0]: dict(zip(headers, values[0:])) for values in schema}