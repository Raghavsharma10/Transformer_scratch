def import_tables(self, only_tables=None, exclude_tables=None):
        """Imports all data in database tables

        :param set[str] only_tables: names of tables to be imported
        :param set[str] exclude_tables: names of tables to be excluded
        """
        for table in self.tables:
            if only_tables is not None and table.name not in only_tables:
                continue

            if exclude_tables is not None and table.name in exclude_tables:
                continue
            self.import_table(table)