def _get_select_commands(self, source, tables):
        """
        Create select queries for all of the tables from a source database.

        :param source: Source database name
        :param tables: Iterable of table names
        :return: Dictionary of table keys, command values
        """
        # Create dictionary of select queries
        row_queries = {tbl: self.select_all(tbl, execute=False) for tbl in
                       tqdm(tables, total=len(tables), desc='Getting {0} select queries'.format(source))}

        # Convert command strings into lists of commands
        for tbl, command in row_queries.items():
            if isinstance(command, str):
                row_queries[tbl] = [command]

        # Pack commands into list of tuples
        return [(tbl, cmd) for tbl, cmds in row_queries.items() for cmd in cmds]