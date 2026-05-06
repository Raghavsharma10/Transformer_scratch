def p_switch_statement(self, p):
        """switch_statement : SWITCH LPAREN expr RPAREN case_block"""
        cases = []
        default = None
        # iterate over return values from case_block
        for item in p[5]:
            if isinstance(item, ast.Default):
                default = item
            elif isinstance(item, list):
                cases.extend(item)

        p[0] = ast.Switch(expr=p[3], cases=cases, default=default)