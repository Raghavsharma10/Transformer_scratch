def p_try_statement_1(self, p):
        """try_statement : TRY block catch"""
        p[0] = ast.Try(statements=p[2], catch=p[3])