def default(self, request, tag):
        """
        Render the initial value of the wrapped L{Parameter} instance.
        """
        if self.parameter.default is not None:
            tag[self.parameter.default]
        return tag