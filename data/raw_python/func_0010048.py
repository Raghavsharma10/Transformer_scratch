def check_email_exists_by_subject(self, subject, match_recipient=None):
        """
        Searches for Email by Subject.  Returns True or False.

        Args:
            subject (str): Subject to search for.

        Kwargs:
            match_recipient (str) : Recipient to match exactly. (don't care if not specified)

        Returns: 
            True - email found, False - email not found

        """
        # Select inbox to fetch the latest mail on server.
        self._mail.select("inbox")

        try:
            matches = self.__search_email_by_subject(subject, match_recipient)
            if len(matches) <= 0:
                return False
            else:
                return True
        except Exception as e:
            raise e