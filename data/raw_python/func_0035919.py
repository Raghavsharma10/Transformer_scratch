def delete_attachment(cls, session, attachment):
        """Delete an attachment.

        Args:
            session (requests.sessions.Session): Authenticated session.
            attachment (helpscout.models.Attachment): The attachment to
                be deleted.

        Returns:
            NoneType: Nothing.
        """
        return super(Conversations, cls).delete(
            session,
            attachment,
            endpoint_override='/attachments/%s.json' % attachment.id,
            out_type=Attachment,
        )