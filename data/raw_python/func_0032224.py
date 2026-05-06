def editContactItem(self, contactType, contactItem, contactInfo):
        """
        Edit the given contact item with the given contact type.  Broadcast
        the edit to all L{IOrganizerPlugin} powerups.

        @type contactType: L{IContactType}
        @param contactType: The contact type which will be used to edit the
            contact item.

        @param contactItem: The contact item to edit.

        @type contactInfo: C{dict}
        @param contactInfo: The contact information to use to edit the
            contact item.

        @return: C{None}
        """
        contactType.editContactItem(
            contactItem, **_stringifyKeys(contactInfo))
        self._callOnOrganizerPlugins('contactItemEdited', contactItem)