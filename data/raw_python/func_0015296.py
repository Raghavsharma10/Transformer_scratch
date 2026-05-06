def symbol(self, id, bp=0):
        """
        Adds symbol 'id' to symbol_table if it does not exist already,
        if it does it merely updates its binding power and returns it's
        symbol class
        """

        try:
            s = self.symbol_table[id]
        except KeyError:
            class s(self.symbol_base):
                pass
            s.id = id
            s.lbp = bp
            self.symbol_table[id] = s
        else:
            s.lbp = max(bp, s.lbp)
        return s