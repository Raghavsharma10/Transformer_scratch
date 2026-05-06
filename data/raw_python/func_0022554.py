def dot_rhs(self):
        """Match the right-hand side of a dot (.) operator.

        The RHS must be a symbol token, but it is interpreted as a literal
        string (because that's what goes in the AST of Resolve.)
        """
        self.tokens.expect(common_grammar.symbol)
        return ast.Literal(self.tokens.matched.value,
                           start=self.tokens.matched.start,
                           end=self.tokens.matched.end, source=self.original)