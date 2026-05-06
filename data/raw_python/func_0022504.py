def accept(self, f, *args):
        """Like 'match', but consume the token (tokenizer advances.)"""
        match = self.match(f, *args)
        if match is None:
            return

        self.tokenizer.skip(len(match.tokens))
        return match