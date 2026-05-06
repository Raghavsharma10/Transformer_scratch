def teardown(self):
        """Cleanup cache tables."""
        for table_spec in reversed(self._table_specs):
            with self._conn:
                table_spec.teardown(self._conn)