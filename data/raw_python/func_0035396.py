def p_postfix_expr(self, p):
        """postfix_expr : left_hand_side_expr
                        | left_hand_side_expr PLUSPLUS
                        | left_hand_side_expr MINUSMINUS
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = self.asttypes.PostfixExpr(op=p[2], value=p[1])
            p[0].setpos(p, 2)