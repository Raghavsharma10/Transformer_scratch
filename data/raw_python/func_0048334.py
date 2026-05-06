def p_assignment_expr(self, p):
        """
        assignment_expr \
            : conditional_expr
            | left_hand_side_expr assignment_operator assignment_expr
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = ast.Assign(left=p[1], op=p[2], right=p[3])