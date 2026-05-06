def array_items(self, number_type, *, number_suffix=''):
        """Parse and yield array items from the token stream."""
        for token in self.collect_tokens_until('CLOSE_BRACKET'):
            is_number = token.type == 'NUMBER'
            value = token.value.lower()
            if not (is_number and value.endswith(number_suffix)):
                raise self.error(f'Invalid {number_type} array element '
                                 f'{token.value!r}')
            yield int(value.replace(number_suffix, ''))