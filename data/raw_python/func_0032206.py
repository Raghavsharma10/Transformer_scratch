def createContactItem(self, person, email):
        """
        Create a new L{EmailAddress} associated with the given person based on
        the given email address.

        @type person: L{Person}
        @param person: The person with whom to associate the new
            L{EmailAddress}.

        @type email: C{unicode}
        @param email: The value to use for the I{address} attribute of the
            newly created L{EmailAddress}.  If C{''}, no L{EmailAddress} will
            be created.

        @return: C{None}
        """
        if email:
            address = self._existing(email)
            if address is not None:
                raise ValueError('There is already a person with that email '
                                 'address (%s): ' % (address.person.name,))
            return EmailAddress(store=self.store,
                                address=email,
                                person=person)