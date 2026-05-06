def _check_symbols(self, symbols):
        """the size must be the same as the length of the array numbers and all elements must be strings"""
        if len(symbols) != self.size:
            raise TypeError("The number of symbols in the graph does not "
                "match the length of the atomic numbers array.")
        for symbol in symbols:
            if not isinstance(symbol, str):
                raise TypeError("All symbols must be strings.")