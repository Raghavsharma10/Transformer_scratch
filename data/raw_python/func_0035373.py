def p_boolean_literal(self, p):
        """boolean_literal : TRUE
                           | FALSE
        """
        p[0] = self.asttypes.Boolean(p[1])
        p[0].setpos(p)