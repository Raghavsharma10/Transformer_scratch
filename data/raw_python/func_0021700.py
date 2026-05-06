def collect_tokens_until(self, token_type):
        """Yield the item tokens in a comma-separated tag collection."""
        self.next()
        if self.current_token.type == token_type:
            return

        while True:
            yield self.current_token

            self.next()
            if self.current_token.type == token_type:
                return

            if self.current_token.type != 'COMMA':
                raise self.error(f'Expected comma but got '
                                 f'{self.current_token.value!r}')
            self.next()