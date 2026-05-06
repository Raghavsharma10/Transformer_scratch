def p_labelled_statement(self, p):
        """labelled_statement : identifier COLON statement"""
        p[0] = ast.Label(identifier=p[1], statement=p[3])