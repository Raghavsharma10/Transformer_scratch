def createContactItem(self, person, notes):
        """
        Create a new L{Notes} associated with the given person based on the
        given string.

        @type person: L{Person}
        @param person: The person with whom to associate the new L{Notes}.

        @type notes: C{unicode}
        @param notes: The value to use for the I{notes} attribute of the newly
        created L{Notes}.  If C{''}, no L{Notes} will be created.

        @rtype: L{Notes} or C{NoneType}
        """
        if notes:
            return Notes(
                store=person.store, person=person, notes=notes)