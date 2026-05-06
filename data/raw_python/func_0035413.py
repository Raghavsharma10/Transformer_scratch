def p_iteration_statement_3(self, p):
        """
        iteration_statement \
            : FOR LPAREN expr_noin_opt SEMI expr_opt SEMI expr_opt RPAREN \
                  statement
            | FOR LPAREN VAR variable_declaration_list_noin SEMI expr_opt SEMI\
                  expr_opt RPAREN statement
        """
        def wrap(node, key):
            if node is None:
                # work around bug with yacc tracking of empty elements
                # by using the previous token, and increment the
                # positions
                node = self.asttypes.EmptyStatement(';')
                node.setpos(p, key - 1)
                node.lexpos += 1
                node.colno += 1
            else:
                node = self.asttypes.ExprStatement(expr=node)
                node.setpos(p, key)
            return node

        if len(p) == 10:
            p[0] = self.asttypes.For(
                init=wrap(p[3], 3), cond=wrap(p[5], 5),
                count=p[7], statement=p[9])
        else:
            init = self.asttypes.VarStatement(p[4])
            init.setpos(p, 3)
            p[0] = self.asttypes.For(
                init=init, cond=wrap(p[6], 6), count=p[8], statement=p[10])
        p[0].setpos(p)