def p_case_clause(self, p):
        """case_clause : CASE expr COLON source_elements"""
        p[0] = self.asttypes.Case(expr=p[2], elements=p[4])
        p[0].setpos(p)