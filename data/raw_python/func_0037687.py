def _execute_select_commands(self, source, commands):
        """Execute select queries for all of the tables from a source database."""
        rows = {}
        for tbl, command in tqdm(commands, total=len(commands), desc='Executing {0} select queries'.format(source)):
            # Add key to dictionary
            if tbl not in rows:
                rows[tbl] = []
            rows[tbl].extend(self.fetch(command, commit=True))
        self._commit()
        return rows