def p_function_declaration(self, p):
        """
        function_declaration \
            : FUNCTION identifier LPAREN RPAREN LBRACE function_body RBRACE
            | FUNCTION identifier LPAREN formal_parameter_list RPAREN LBRACE \
                 function_body RBRACE
        """
        if len(p) == 8:
            p[0] = self.asttypes.FuncDecl(
                identifier=p[2], parameters=None, elements=p[6])
        else:
            p[0] = self.asttypes.FuncDecl(
                identifier=p[2], parameters=p[4], elements=p[7])
        p[0].setpos(p)