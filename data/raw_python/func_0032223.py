def createContactItem(self, contactType, person, contactInfo):
        """
        Create a new contact item for the given person with the given contact
        type.  Broadcast a creation to all L{IOrganizerPlugin} powerups.

        @type contactType: L{IContactType}
        @param contactType: The contact type which will be used to create the
            contact item.

        @type person: L{Person}
        @param person: The person with whom the contact item will be
            associated.

        @type contactInfo: C{dict}
        @param contactInfo: The contact information to use to create the
            contact item.

        @return: The contact item, as created by the given contact type.
        """
        contactItem = contactType.createContactItem(
            person, **_stringifyKeys(contactInfo))
        if contactItem is not None:
            self._callOnOrganizerPlugins('contactItemCreated', contactItem)
        return contactItem