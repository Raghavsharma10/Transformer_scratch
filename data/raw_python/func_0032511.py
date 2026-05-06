def label(self, request, tag):
        """
        Render the label of the wrapped L{Parameter} or L{ChoiceParameter} instance.
        """
        if self.parameter.label:
            tag[self.parameter.label]
        return tag