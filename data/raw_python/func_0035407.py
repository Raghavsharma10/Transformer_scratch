def p_empty_statement(self, p):
        """empty_statement : SEMI"""
        p[0] = self.asttypes.EmptyStatement(p[1])
        p[0].setpos(p)