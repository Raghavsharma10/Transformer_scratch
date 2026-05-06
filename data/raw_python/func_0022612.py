def next_token(self):
        """Returns the next logical token, advancing the tokenizer."""
        if self.lookahead:
            self.current_token = self.lookahead.popleft()
            return self.current_token

        self.current_token = self._parse_next_token()
        return self.current_token