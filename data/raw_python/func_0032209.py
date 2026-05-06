def getEmailAddresses(self):
        """
        Return an iterator of all email addresses associated with this person.

        @return: an iterator of unicode strings in RFC2822 address format.
        """
        return self.store.query(
            EmailAddress,
            EmailAddress.person == self).getColumn('address')