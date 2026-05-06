def list(self):
        """Parse a list (tuple) which can contain any combination of types."""
        start = self.tokens.matched.start

        if self.tokens.accept(common_grammar.rbracket):
            return ast.Tuple(start=start, end=self.tokens.matched.end,
                             source=self.original)

        elements = [self.expression()]

        while self.tokens.accept(common_grammar.comma):
            elements.append(self.expression())

        self.tokens.expect(common_grammar.rbracket)
        return ast.Tuple(*elements, start=start, end=self.tokens.matched.end,
                         source=self.original)