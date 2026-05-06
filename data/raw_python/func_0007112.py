def dumps(self):
        """Return a dictionnary of current tables"""
        return {table_name: getattr(self, table_name).dumps() for table_name in self.TABLES}