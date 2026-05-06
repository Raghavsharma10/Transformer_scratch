def builtin(self, keyword):
        """Parse the pseudo-function application subgrammar."""
        # The match includes the lparen token, so the keyword is just the first
        # token in the match, not the whole thing.
        keyword_start = self.tokens.matched.first.start
        keyword_end = self.tokens.matched.first.end
        self.tokens.expect(common_grammar.lparen)

        if self.tokens.matched.start != keyword_end:
            return self.error(
                "No whitespace allowed between function and lparen.",
                start_token=self.tokens.matched.first)

        expr_type = grammar.BUILTINS[keyword.lower()]
        arguments = [self.expression()]
        while self.tokens.accept(common_grammar.comma):
            arguments.append(self.expression())

        self.tokens.expect(common_grammar.rparen)

        if expr_type.arity and expr_type.arity != len(arguments):
            return self.error(
                "%s expects %d arguments, but was passed %d." % (
                    keyword, expr_type.arity, len(arguments)),
                start_token=self.tokens.matched.first)

        return expr_type(*arguments, start=keyword_start,
                         end=self.tokens.matched.end, source=self.original)