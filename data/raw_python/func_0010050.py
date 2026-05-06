def get_email_message(self, message_uid, message_type="text/plain"):
        """
        Fetch contents of email.

        Args:
            message_uid (int): IMAP Message UID number.

        Kwargs:
            message_type: Can be 'text' or 'html'

        """
        self._mail.select("inbox")
        result = self._mail.uid('fetch', message_uid, "(RFC822)")
        msg = email.message_from_string(result[1][0][1])

        try:
            # Try to handle as multipart message first.
            for part in msg.walk():
                if part.get_content_type() == message_type:
                    return part.get_payload(decode=True)
        except:
            # handle as plain text email
            return msg.get_payload(decode=True)