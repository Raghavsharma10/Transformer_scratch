def skip(self, steps=1):
        """Skip ahead by 'steps' tokens."""
        for _ in six.moves.range(steps):
            self.next_token()