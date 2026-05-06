def compile(self):
        """Compile this expression into a query string"""
        return "{attribute}{sep}{value}".format(
            attribute=self.attribute,
            sep=self.sep,
            value=_quoted(self.value)
        )