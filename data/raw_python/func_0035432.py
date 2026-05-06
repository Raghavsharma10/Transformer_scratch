def p_try_statement_2(self, p):
        """try_statement : TRY block finally"""
        p[0] = self.asttypes.Try(statements=p[2], fin=p[3])
        p[0].setpos(p)