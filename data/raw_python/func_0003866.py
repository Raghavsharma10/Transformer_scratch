def set_default_symbols(self):
        """Set self.symbols based on self.numbers and the periodic table."""
        self.symbols = tuple(periodic[n].symbol for n in self.numbers)