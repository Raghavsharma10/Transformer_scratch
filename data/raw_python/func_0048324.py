def p_property_assignment(self, p):
        """property_assignment \
             : property_name COLON assignment_expr
             | GETPROP property_name LPAREN RPAREN LBRACE function_body RBRACE
             | SETPROP property_name LPAREN formal_parameter_list RPAREN \
                   LBRACE function_body RBRACE
        """
        if len(p) == 4:
            p[0] = ast.Assign(left=p[1], op=p[2], right=p[3])
        elif len(p) == 8:
            p[0] = ast.GetPropAssign(prop_name=p[2], elements=p[6])
        else:
            p[0] = ast.SetPropAssign(
                prop_name=p[2], parameters=p[4], elements=p[7])