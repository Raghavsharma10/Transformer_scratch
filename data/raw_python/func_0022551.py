def atom(self):
        """Parse an atom, which is most things.

        Grammar:
            atom =
                [ prefix ]
                ( select_expression
                | any_expression
                | func_application
                | let_expr
                | var
                | literal
                | list
                | "(" expression ")" ) .
        """
        # Parameter replacement with literals.
        if self.tokens.accept(grammar.param):
            return self.param()

        # Let expressions (let(x = 5, y = 10) x + y)
        if self.tokens.accept(grammar.let):
            return self.let()

        # At the top level, we try to see if we are recursing into an SQL query.
        if self.tokens.accept(grammar.select):
            return self.select()

        # A SELECT query can also start with 'ANY'.
        if self.tokens.accept(grammar.select_any):
            return self.select_any()

        # Explicitly reject any keywords from SQL other than SELECT and ANY.
        # If we don't do this they will match as valid symbols (variables)
        # and that might be confusing to the user.
        self.tokens.reject(grammar.sql_keyword)

        # Match if-else before other things that consume symbols.
        if self.tokens.accept(grammar.if_if):
            return self.if_if()

        # Operators must be matched first because the same symbols could also
        # be vars or applications.
        if self.tokens.accept(grammar.prefix):
            operator = self.tokens.matched.operator
            start = self.tokens.matched.start
            expr = self.expression(operator.precedence)
            return operator.handler(expr, start=start, end=expr.end,
                                    source=self.original)

        if self.tokens.accept(grammar.literal):
            return ast.Literal(self.tokens.matched.value, source=self.original,
                               start=self.tokens.matched.start,
                               end=self.tokens.matched.end)

        # Match builtin pseudo-functions before functions and vars to prevent
        # overrides.
        if self.tokens.accept(grammar.builtin):
            return self.builtin(self.tokens.matched.value)

        # Match applications before vars, because obviously.
        if self.tokens.accept(grammar.application):
            return self.application(
                ast.Var(self.tokens.matched.value, source=self.original,
                        start=self.tokens.matched.start,
                        end=self.tokens.matched.first.end))

        if self.tokens.accept(common_grammar.symbol):
            return ast.Var(self.tokens.matched.value, source=self.original,
                           start=self.tokens.matched.start,
                           end=self.tokens.matched.end)

        if self.tokens.accept(common_grammar.lparen):
            # Parens will contain one or more expressions. If there are several
            # expressions, separated by commas, then they are a repeated value.
            #
            # Unlike lists, repeated values must all be of the same type,
            # otherwise evaluation of the query will fail at runtime (or
            # type-check time, for simple cases.)
            start = self.tokens.matched.start
            expressions = [self.expression()]

            while self.tokens.accept(common_grammar.comma):
                expressions.append(self.expression())

            self.tokens.expect(common_grammar.rparen)

            if len(expressions) == 1:
                return expressions[0]
            else:
                return ast.Repeat(*expressions, source=self.original,
                                  start=start, end=self.tokens.matched.end)

        if self.tokens.accept(common_grammar.lbracket):
            return self.list()

        # We've run out of things we know the next atom could be. If there is
        # still input left then it's illegal syntax. If there is nothing then
        # the input cuts off when we still need an atom. Either is an error.
        if self.tokens.peek(0):
            return self.error(
                "Was not expecting %r here." % self.tokens.peek(0).name,
                start_token=self.tokens.peek(0))
        else:
            return self.error("Unexpected end of input.")