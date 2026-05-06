def _sort_lines(self):
        """Haproxy writes its logs after having gathered all information
        related to each specific connection. A simple request can be
        really quick but others can be really slow, thus even if one connection
        is logged later, it could have been accepted before others that are
        already processed and logged.

        This method sorts all valid log lines by their acceptance date,
        providing the real order in which connections where made to the server.
        """
        self._valid_lines = sorted(
            self._valid_lines,
            key=lambda line: line.accept_date,
        )