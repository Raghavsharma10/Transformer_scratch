def editContactItem(self, contact, email):
        """
        Change the email address of the given L{EmailAddress} to that specified
        by C{email}.

        @type email: C{unicode}
        @param email: The new value to use for the I{address} attribute of the
            L{EmailAddress}.

        @return: C{None}
        """
        address = self._existing(email)
        if address is not None and address is not contact:
            raise ValueError('There is already a person with that email '
                             'address (%s): ' % (address.person.name,))
        contact.address = email