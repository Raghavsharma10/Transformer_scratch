def _select_batched(self, table, cols, num_rows, limit, queries_per_batch=3, execute=True):
        """Run select queries in small batches and return joined resutls."""
        # Execute select queries in small batches to avoid connection timeout
        commands, offset = [], 0
        while num_rows > 0:
            # Use number of rows as limit if num_rows < limit
            _limit = min(limit, num_rows)

            # Execute select_limit query
            commands.append(self._select_limit_statement(table, cols=cols, offset=offset, limit=limit))
            offset += _limit
            num_rows += -_limit

        # Execute commands
        if execute:
            rows = []
            til_reconnect = queries_per_batch
            for c in commands:
                if til_reconnect == 0:
                    self.disconnect()
                    self.reconnect()
                    til_reconnect = queries_per_batch
                rows.extend(self.fetch(c, False))
                til_reconnect += -1
            del commands
            return rows
        # Return commands
        else:
            return commands