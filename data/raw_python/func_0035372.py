def p_block(self, p):
        """block : LBRACE source_elements RBRACE"""
        p[0] = self.asttypes.Block(p[2])
        p[0].setpos(p)