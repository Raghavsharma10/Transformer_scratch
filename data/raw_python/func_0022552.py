def accept_operator(self, precedence):
        """Accept the next binary operator only if it's of higher precedence."""
        match = grammar.infix(self.tokens)
        if not match:
            return

        if match.operator.precedence < precedence:
            return

        # The next thing is an operator that we want. Now match it for real.
        return self.tokens.accept(grammar.infix)