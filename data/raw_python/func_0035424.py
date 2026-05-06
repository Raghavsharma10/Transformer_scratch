def p_switch_statement(self, p):
        """switch_statement : SWITCH LPAREN expr RPAREN case_block"""
        # this uses a completely different type that corrects a
        # subtly wrong interpretation of this construct.
        # see: https://github.com/rspivak/slimit/issues/94
        p[0] = self.asttypes.Switch(expr=p[3], case_block=p[5])
        p[0].setpos(p)
        return