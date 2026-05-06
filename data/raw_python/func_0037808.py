def drop_primary_key(self, table):
        """Drop a Primary Key constraint for a specific table."""
        if self.get_primary_key(table):
            self.execute('ALTER TABLE {0} DROP PRIMARY KEY'.format(wrap(table)))