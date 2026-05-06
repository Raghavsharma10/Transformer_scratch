def p_iteration_statement_4(self, p):
        """
        iteration_statement \
            : FOR LPAREN left_hand_side_expr IN expr RPAREN statement
        """
        p[0] = self.asttypes.ForIn(item=p[3], iterable=p[5], statement=p[7])
        p[0].setpos(p)