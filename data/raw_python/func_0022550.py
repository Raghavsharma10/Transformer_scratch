def expression(self, previous_precedence=0):
        """An expression is an atom or an infix expression.

        Grammar (sort of, actually a precedence-climbing parser):
            expression = atom [ binary_operator expression ] .

        Args:
            previous_precedence: What operator precedence should we start with?
        """
        lhs = self.atom()

        return self.operator(lhs, previous_precedence)