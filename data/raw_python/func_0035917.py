def create_attachment(cls, session, attachment):
        """Create an attachment.

        An attachment must be sent to the API before it can be used in a
        thread. Use this method to create the attachment, then use the
        resulting hash when creating a thread.

        Note that HelpScout only supports attachments of 10MB or lower.

        Args:
            session (requests.sessions.Session): Authenticated session.
            attachment (helpscout.models.Attachment): The attachment to be
             created.

        Returns:
            helpscout.models.Attachment: The newly created attachment (hash
             property only). Use this hash when associating the attachment with
             a new thread.
        """
        return super(Conversations, cls).create(
            session,
            attachment,
            endpoint_override='/attachments.json',
            out_type=Attachment,
        )