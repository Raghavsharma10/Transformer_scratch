def makeEditorialLiveForms(self):
        """
        Make some L{liveform.LiveForm} instances for editing the contact
        information of the wrapped L{Person}.
        """
        parameters = [
            liveform.Parameter(
                'nickname', liveform.TEXT_INPUT,
                _normalizeWhitespace, 'Name',
                default=self.person.name)]
        separateForms = []
        for contactType in self.organizer.getContactTypes():
            if getattr(contactType, 'getEditFormForPerson', None):
                editForm = contactType.getEditFormForPerson(self.person)
                if editForm is not None:
                    editForm.setFragmentParent(self)
                    separateForms.append(editForm)
                    continue
            param = self.organizer.toContactEditorialParameter(
                contactType, self.person)
            parameters.append(param)
            self.contactTypes[param.name] = contactType
        form = liveform.LiveForm(
            self.editContactItems, parameters, u'Save')
        form.compact()
        form.jsClass = u'Mantissa.People.EditPersonForm'
        form.setFragmentParent(self)
        return [form] + separateForms