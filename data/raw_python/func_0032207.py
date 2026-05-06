def getContactItems(self, person):
        """
        Return all L{EmailAddress} instances associated with the given person.

        @type person: L{Person}
        """
        return person.store.query(
            EmailAddress,
            EmailAddress.person == person)