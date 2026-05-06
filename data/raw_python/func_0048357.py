def p_function_expr_1(self, p):
        """
        function_expr \
            : FUNCTION LPAREN RPAREN LBRACE function_body RBRACE
            | FUNCTION LPAREN formal_parameter_list RPAREN \
                LBRACE function_body RBRACE
        """
        if len(p) == 7:
            p[0] = ast.FuncExpr(
                identifier=None, parameters=None, elements=p[5])
        else:
            p[0] = ast.FuncExpr(
                identifier=None, parameters=p[3], elements=p[6])