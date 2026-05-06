def _initialize_tables(self):
        """Create tables for structure and values, word->vocabulary"""
        # structure table
        self.table_struct, self.idnt_struct_size = self._create_struct_table()
        # values table
        self.table_values, self.idnt_values_size = self._create_values_table()