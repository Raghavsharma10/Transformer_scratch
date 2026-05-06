def description(self, request, tag):
        """
        Render the description of the wrapped L{Parameter} instance.
        """
        if self.parameter.description is not None:
            tag[self.parameter.description]
        return tag