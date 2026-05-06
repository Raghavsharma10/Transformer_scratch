def p_assignment_expr_noin(self, p):
        """
        assignment_expr_noin \
            : conditional_expr_noin
            | left_hand_side_expr assignment_operator assignment_expr_noin
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = self.asttypes.Assign(left=p[1], op=p[2], right=p[3])
            p[0].setpos(p, 2)