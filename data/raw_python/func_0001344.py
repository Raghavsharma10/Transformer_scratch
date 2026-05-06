def create_column(self, name, type):
        """
        Explicitely create a new column ``name`` of a specified type.
        ``type`` must be a `SQLAlchemy column type <http://docs.sqlalchemy.org/en/rel_0_8/core/types.html>`_.
        ::

            table.create_column('created_at', sqlalchemy.DateTime)
        """
        self._check_dropped()
        if normalize_column_name(name) not in self._normalized_columns:
            self.op.add_column(self.table.name, Column(name, type), self.table.schema)
            self.table = self._update_table(self.table.name)