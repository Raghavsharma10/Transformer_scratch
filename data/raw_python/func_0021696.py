def next(self):
        """Move to the next token in the token stream."""
        self.current_token = next(self.token_stream, None)
        if self.current_token is None:
            self.token_span = self.token_span[1], self.token_span[1]
            raise self.error('Unexpected end of input')
        self.token_span = self.current_token.span
        return self