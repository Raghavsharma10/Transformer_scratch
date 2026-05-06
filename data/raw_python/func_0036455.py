def get_contact(self, email):
        """Get Filemail contact based on email.

        :param email: address of contact
        :type email: ``str``, ``unicode``
        :rtype: ``dict`` with contact information
        """

        contacts = self.get_contacts()
        for contact in contacts:
            if contact['email'] == email:
                return contact

        msg = 'No contact with email: "{email}" found.'
        raise FMBaseError(msg.format(email=email))