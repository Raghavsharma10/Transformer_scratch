def _parse_next_token(self):
        """Will parse patterns until it gets to the next token or EOF."""
        while self._position < self.limit:
            token = self._next_pattern()
            if token:
                return token

        return None