def _copy_database_data_serverside(self, source, destination, tables):
        """Select rows from a source database and insert them into a destination db in one query"""
        for table in tqdm(tables, total=len(tables), desc='Copying table data (optimized)'):
            self.execute('INSERT INTO {0}.{1} SELECT * FROM {2}.{1}'.format(destination, wrap(table), source))