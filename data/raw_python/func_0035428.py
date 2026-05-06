def p_default_clause(self, p):
        """default_clause : DEFAULT COLON source_elements"""
        p[0] = self.asttypes.Default(elements=p[3])
        p[0].setpos(p)