def getContactItems(self, person):
        """
        Return an iterable of L{PhoneNumber} items that are associated with
        C{person}.

        @type person: L{Person}
        """
        return person.store.query(
            PhoneNumber, PhoneNumber.person == person)