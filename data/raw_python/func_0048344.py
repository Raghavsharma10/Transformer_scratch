def p_iteration_statement_3(self, p):
        """
        iteration_statement \
            : FOR LPAREN expr_noin_opt SEMI expr_opt SEMI expr_opt RPAREN \
                  statement
            | FOR LPAREN VAR variable_declaration_list_noin SEMI expr_opt SEMI\
                  expr_opt RPAREN statement
        """
        if len(p) == 10:
            p[0] = ast.For(init=p[3], cond=p[5], count=p[7], statement=p[9])
        else:
            init = ast.VarStatement(p[4])
            p[0] = ast.For(init=init, cond=p[6], count=p[8], statement=p[10])