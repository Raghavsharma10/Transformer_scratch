def parse_compound(self):
        """Parse a compound from the token stream."""
        compound_tag = Compound()

        for token in self.collect_tokens_until('CLOSE_COMPOUND'):
            item_key = token.value
            if token.type not in ('NUMBER', 'STRING', 'QUOTED_STRING'):
                raise self.error(f'Expected compound key but got {item_key!r}')

            if token.type == 'QUOTED_STRING':
                item_key = self.unquote_string(item_key)

            if self.next().current_token.type != 'COLON':
                raise self.error(f'Expected colon but got '
                                 f'{self.current_token.value!r}')
            self.next()
            compound_tag[item_key] = self.parse()
        return compound_tag