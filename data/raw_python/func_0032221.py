def getContactEditorialParameters(self, person):
        """
        Yield L{LiveForm} parameters to edit each contact item of each contact
        type for the given person.

        @type person: L{Person}
        @return: An iterable of two-tuples.  The first element of each tuple
            is an L{IContactType} provider.  The third element of each tuple
            is the L{LiveForm} parameter object for that contact item.
        """
        for contactType in self.getContactTypes():
            yield (
                contactType,
                self.toContactEditorialParameter(contactType, person))