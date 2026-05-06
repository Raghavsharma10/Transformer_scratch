def drop_column(self, name):
        """
        Drop the column ``name``
        ::

            table.drop_column('created_at')
        """
        self._check_dropped()
        if name in list(self.table.columns.keys()):
            self.op.drop_column(self.table.name, name, schema=self.schema)
            self.table = self._update_table(self.table.name)