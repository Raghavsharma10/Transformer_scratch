def reference(self, symbol, count=1):
        """
        However, if referenced, ensure that the counter is applied to
        the catch symbol.
        """

        if symbol == self.catch_symbol:
            self.catch_symbol_usage += count
        else:
            self.parent.reference(symbol, count)