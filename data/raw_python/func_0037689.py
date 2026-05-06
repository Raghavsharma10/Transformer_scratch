def _get_insert_commands(self, rows, cols):
        """Retrieve dictionary of insert statements to be executed."""
        # Get insert queries
        insert_queries = {}
        for table in tqdm(list(rows.keys()), total=len(list(rows.keys())), desc='Getting insert rows queries'):
            insert_queries[table] = {}
            _rows = rows.pop(table)
            _cols = cols.pop(table)

            if len(_rows) > 1:
                insert_queries[table]['insert_many'] = self.insert_many(table, _cols, _rows, execute=False)
            elif len(_rows) == 1:
                insert_queries[table]['insert'] = self.insert(table, _cols, _rows, execute=False)
        return insert_queries