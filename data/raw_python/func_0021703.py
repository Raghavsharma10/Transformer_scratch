def parse_list(self):
        """Parse a list from the token stream."""
        try:
            return List([self.parse() for _ in
                         self.collect_tokens_until('CLOSE_BRACKET')])
        except IncompatibleItemType as exc:
            raise self.error(f'Item {str(exc.item)!r} is not a '
                             f'{exc.subtype.__name__} tag') from None