def setup(self):
        """Setup cache tables."""
        for table_spec in self._table_specs:
            with self._conn:
                table_spec.setup(self._conn)