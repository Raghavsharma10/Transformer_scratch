def p_catch(self, p):
        """catch : CATCH LPAREN identifier RPAREN block"""
        p[0] = ast.Catch(identifier=p[3], elements=p[5])