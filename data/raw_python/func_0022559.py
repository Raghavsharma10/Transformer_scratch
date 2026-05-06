def application(self, func):
        """Parse the function application subgrammar.

        Function application can, conceptually, be thought of as a mixfix
        operator, similar to the way array subscripting works. However, it is
        not clear at this point whether we want to allow it to work as such,
        because doing so would permit queries to, at runtime, select methods
        out of an arbitrary object and then call them.

        While there is a function whitelist and preventing this sort of thing
        in the syntax isn't a security feature, it still seems like the
        syntax should make it clear what the intended use of application is.

        If we later decide to extend DottySQL to allow function application
        over an arbitrary LHS expression then that syntax would be a strict
        superset of the current syntax and backwards compatible.
        """
        start = self.tokens.matched.start
        if self.tokens.accept(common_grammar.rparen):
            # That was easy.
            return ast.Apply(func, start=start, end=self.tokens.matched.end,
                             source=self.original)

        arguments = [self.expression()]
        while self.tokens.accept(common_grammar.comma):
            arguments.append(self.expression())

        self.tokens.expect(common_grammar.rparen)
        return ast.Apply(func, *arguments, start=start,
                         end=self.tokens.matched.end, source=self.original)