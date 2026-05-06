def table(self):
        """a ```pylsdj.Table``` referencing the instrument's table, or None
        if the instrument doesn't have a table"""
        if hasattr(self.data, 'table_on') and self.data.table_on:
            assert_index_sane(self.data.table, len(self.song.tables))
            return self.song.tables[self.data.table]