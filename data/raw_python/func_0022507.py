def current_position(self):
        """Return a tuple of (start, end)."""
        token = self.tokenizer.peek(0)
        if token:
            return token.start, token.end

        return self.tokenizer.position, self.tokenizer.position + 1