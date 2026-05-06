def getContactItems(self, person):
        """
        Return a C{list} of the L{PostalAddress} items associated with the
        given person.

        @type person: L{Person}
        """
        return person.store.query(PostalAddress, PostalAddress.person == person)