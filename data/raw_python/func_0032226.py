def editPerson(self, person, nickname, edits):
        """
        Change the name and contact information associated with the given
        L{Person}.

        @type person: L{Person}
        @param person: The person which will be modified.

        @type nickname: C{unicode}
        @param nickname: The new value for L{Person.name}

        @type edits: C{list}
        @param edits: list of tuples of L{IContactType} providers and
        corresponding L{ListChanges} objects or dictionaries of parameter
        values.
        """
        for existing in self.store.query(Person, Person.name == nickname):
            if existing is person:
                continue
            raise ValueError(
                "A person with the name %r exists already." % (nickname,))
        oldname = person.name
        person.name = nickname
        self._callOnOrganizerPlugins('personNameChanged', person, oldname)
        for contactType, submission in edits:
            if contactType.allowMultipleContactItems:
                for edit in submission.edit:
                    self.editContactItem(
                        contactType, edit.object, edit.values)
                for create in submission.create:
                    create.setter(
                        self.createContactItem(
                            contactType, person, create.values))
                for delete in submission.delete:
                    delete.deleteFromStore()
            else:
                (contactItem,) = contactType.getContactItems(person)
                self.editContactItem(
                    contactType, contactItem, submission)