def selected(self, request, tag):
        """
        Render a selected attribute on the given tag if the wrapped L{Option}
        instance is selected.
        """
        if self.option.selected:
            tag(selected='selected')
        return tag