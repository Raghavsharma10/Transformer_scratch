def compile(self):
        """Compile this expression into a query string"""
        return "{lhs}{sep}{rhs}".format(
            lhs=self.lhs.compile(),
            sep=self.sep,
            rhs=self.rhs.compile(),
        )