def peek(self, steps=1):
        """Look ahead, doesn't affect current_token and next_token."""
        try:
            tokens = iter(self)
            for _ in six.moves.range(steps):
                next(tokens)

            return next(tokens)
        except StopIteration:
            return None