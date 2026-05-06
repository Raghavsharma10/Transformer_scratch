def set_symbol(self, symbol):
        """(symbol, bondorder) -> set the bondsymbol
        of the molecule"""
        raise "Deprecated"
        self.symbol, self.bondtype, bondorder, self.equiv_class = \
                     BONDLOOKUP[symbol]
        if self.bondtype == 4:
            self.aromatic = 1
        else:
            self.aromatic = 0