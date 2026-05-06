def editContactItems(self, nickname, **edits):
        """
        Update the information on the contact items associated with the wrapped
        L{Person}.

        @type nickname: C{unicode}
        @param nickname: New value to use for the I{name} attribute of the
            L{Person}.

        @param **edits: mapping from contact type identifiers to
            ListChanges instances.
        """
        submissions = []
        for paramName, submission in edits.iteritems():
            contactType = self.contactTypes[paramName]
            submissions.append((contactType, submission))
        self.person.store.transact(
            self.organizer.editPerson,
            self.person, nickname, submissions)