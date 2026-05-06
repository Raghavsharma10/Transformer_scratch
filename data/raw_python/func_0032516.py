def multiple(self, request, tag):
        """
        Render a I{multiple} attribute on the given tag if the wrapped
        L{ChoiceParameter} instance allows multiple selection.
        """
        if self.parameter.multiple:
            tag(multiple='multiple')
        return tag