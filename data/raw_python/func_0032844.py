def create(self, tables, version):
        """Do the actual work of creating the database, filling its tables with
        values, creating indices, and setting the datacache version metadata.

        Parameters
        ----------
        tables : list
            List of datacache.DatabaseTable objects

        version : int
        """
        for table in tables:
            self._create_table(
                table_name=table.name,
                column_types=table.column_types,
                primary=table.primary_key,
                nullable=table.nullable)
            self._fill_table(table.name, table.rows)
            self._create_indices(table.name, table.indices)
        self._finalize_database(version)
        self._commit()