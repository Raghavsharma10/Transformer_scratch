def find_emails_by_subject(self, subject, limit=50, match_recipient=None):
        """
        Searches for Email by Subject.  Returns email's imap message IDs 
        as a list if matching subjects is found.

        Args:
            subject (str) - Subject to search for.

        Kwargs:
            limit (int) - Limit search to X number of matches, default 50
            match_recipient (str) - Recipient to exactly (don't care if not specified)

        Returns:
            list - List of Integers representing imap message UIDs.

        """
        # Select inbox to fetch the latest mail on server.
        self._mail.select("inbox")

        matching_uids = self.__search_email_by_subject(
            subject, match_recipient)

        return matching_uids