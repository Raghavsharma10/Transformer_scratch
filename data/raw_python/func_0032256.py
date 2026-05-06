def importPeopleForm(self, request, tag):
        """
        Create and return a L{liveform.LiveForm} for adding new L{Person}s.
        """
        form = liveform.LiveForm(
            self.importAddresses,
            [liveform.Parameter('addresses', liveform.TEXTAREA_INPUT,
                                self._parseAddresses, 'Email Addresses')],
            description='Import People')
        form.jsClass = u'Mantissa.People.ImportPeopleForm'
        form.compact()
        form.setFragmentParent(self)
        return form