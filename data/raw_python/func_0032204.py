def getParameters(self, emailAddress):
        """
        Return a C{list} of one L{LiveForm} parameter for editing an
        L{EmailAddress}.

        @type emailAddress: L{EmailAddress} or C{NoneType}
        @param emailAddress: If not C{None}, an existing contact item from
            which to get the email address default value.

        @rtype: C{list}
        @return: The parameters necessary for specifying an email address.
        """
        if emailAddress is not None:
            address = emailAddress.address
        else:
            address = u''
        return [
            liveform.Parameter('email', liveform.TEXT_INPUT,
                               _normalizeWhitespace, 'Email Address',
                               default=address)]