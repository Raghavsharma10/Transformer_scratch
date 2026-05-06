def input(self, request, tag):
        """
        Add the wrapped form, as a subform, as a child of the given tag.
        """
        subform = self.parameter.form.asSubForm(self.parameter.name)
        subform.setFragmentParent(self)
        return tag[subform]