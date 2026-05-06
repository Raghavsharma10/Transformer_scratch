def editContactItem(self, contact, label, number):
        """
        Change the I{number} attribute of C{contact} to C{number}, and the
        I{label} attribute to C{label}.

        @type contact: L{PhoneNumber}

        @type label: C{unicode}

        @type number: C{unicode}

        @return: C{None}
        """
        contact.label = label
        contact.number = number