def operator(self, lhs, min_precedence):
        """Climb operator precedence as long as there are operators.

        This function implements a basic precedence climbing parser to deal
        with binary operators in a sane fashion. The outer loop will keep
        spinning as long as the next token is an operator with a precedence
        of at least 'min_precedence', parsing operands as atoms (which,
        in turn, recurse into 'expression' which recurses back into 'operator').

        This supports both left- and right-associativity. The only part of the
        code that's not a regular precedence-climber deals with mixfix
        operators. A mixfix operator in DottySQL consists of an infix part
        and a suffix (they are still binary, they just have a terminator).
        """

        # Spin as long as the next token is an operator of higher
        # precedence. (This may not do anything, which is fine.)
        while self.accept_operator(precedence=min_precedence):
            operator = self.tokens.matched.operator

            # If we're parsing a mixfix operator we can keep going until
            # the suffix.
            if operator.suffix:
                rhs = self.expression()
                self.tokens.expect(common_grammar.match_tokens(operator.suffix))
                rhs.end = self.tokens.matched.end
            elif operator.name == ".":
                # The dot operator changes the meaning of RHS.
                rhs = self.dot_rhs()
            else:
                # The right hand side is an atom, which might turn out to be
                # an expression. Isn't recursion exciting?
                rhs = self.atom()

            # Keep going as long as the next token is an infix operator of
            # higher precedence.
            next_min_precedence = operator.precedence
            if operator.assoc == "left":
                next_min_precedence += 1

            while self.tokens.match(grammar.infix):
                if (self.tokens.matched.operator.precedence
                        < next_min_precedence):
                    break
                rhs = self.operator(rhs,
                                    self.tokens.matched.operator.precedence)

            lhs = operator.handler(lhs, rhs, start=lhs.start, end=rhs.end,
                                   source=self.original)

        return lhs