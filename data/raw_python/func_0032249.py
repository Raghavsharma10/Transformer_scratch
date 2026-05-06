def getParameters(self, postalAddress):
        """
        Return a C{list} of one L{LiveForm} parameter for editing a
        L{PostalAddress}.

        @type postalAddress: L{PostalAddress} or C{NoneType}

        @param postalAddress: If not C{None}, an existing contact item from
            which to get the postal address default value.

        @rtype: C{list}
        @return: The parameters necessary for specifying a postal address.
        """
        address = u''
        if postalAddress is not None:
            address = postalAddress.address
        return [
            liveform.Parameter('address', liveform.TEXT_INPUT,
                               unicode, 'Postal Address', default=address)]