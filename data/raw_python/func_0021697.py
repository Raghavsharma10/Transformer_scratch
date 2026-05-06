def parse(self):
        """Parse and return an nbt literal from the token stream."""
        token_type = self.current_token.type.lower()
        handler = getattr(self, f'parse_{token_type}', None)
        if handler is None:
            raise self.error(f'Invalid literal {self.current_token.value!r}')
        return handler()