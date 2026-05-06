def p_new_expr(self, p):
        """new_expr : member_expr
                    | NEW new_expr
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = ast.NewExpr(p[2])