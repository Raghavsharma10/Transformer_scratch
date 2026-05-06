def declare(self, symbol):
        """
        Nothing gets declared here - it's the parents problem, except
        for the case where the symbol is the one we have here.
        """

        if symbol != self.catch_symbol:
            self.parent.declare(symbol)