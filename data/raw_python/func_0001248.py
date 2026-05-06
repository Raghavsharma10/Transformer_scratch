def parse_table_name(self, table):
        """Parse schema qualified table name
        """
        if "." in table:
            schema, table = table.split(".")
        else:
            schema = None
        return (schema, table)