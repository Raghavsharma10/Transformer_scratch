def createContactItem(self, person, address):
        """
        Create a new L{PostalAddress} associated with the given person based on
        the given postal address.

        @type person: L{Person}
        @param person: The person with whom to associate the new
            L{EmailAddress}.

        @type address: C{unicode}
        @param address: The value to use for the I{address} attribute of the
            newly created L{PostalAddress}.  If C{''}, no L{PostalAddress} will
            be created.

        @rtype: L{PostalAddress} or C{NoneType}
        """
        if address:
            return PostalAddress(
                store=person.store, person=person, address=address)