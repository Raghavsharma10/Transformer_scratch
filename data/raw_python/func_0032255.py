def render_addPersonForm(self, ctx, data):
        """
        Create and return a L{liveform.LiveForm} for creating a new L{Person}.
        """
        addPersonForm = liveform.LiveForm(
            self.addPerson, self._baseParameters, description='Add Person')
        addPersonForm.compact()
        addPersonForm.jsClass = u'Mantissa.People.AddPersonForm'
        addPersonForm.setFragmentParent(self)
        return addPersonForm