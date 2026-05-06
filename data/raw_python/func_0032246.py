def createContactItem(self, person, label, number):
        """
        Create a L{PhoneNumber} item for C{number}, associated with C{person}.

        @type person: L{Person}

        @param label: The value to use for the I{label} attribute of the new
        L{PhoneNumber} item.
        @type label: C{unicode}

        @param number: The value to use for the I{number} attribute of the new
        L{PhoneNumber} item.  If C{''}, no item will be created.
        @type number: C{unicode}

        @rtype: L{PhoneNumber} or C{NoneType}
        """
        if number:
            return PhoneNumber(
                store=person.store, person=person, label=label, number=number)