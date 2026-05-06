def p_regex_literal(self, p):
        """regex_literal : REGEX"""
        p[0] = self.asttypes.Regex(p[1])
        p[0].setpos(p)