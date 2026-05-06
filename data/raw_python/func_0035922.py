def get_attachment_data(cls, session, attachment_id):
        """Return a specific attachment's data.

        Args:
            session (requests.sessions.Session): Authenticated session.
            attachment_id (int): The ID of the attachment from which to get
                data.

        Returns:
            helpscout.models.AttachmentData: An attachment data singleton, if
                existing. Otherwise ``None``.
        """
        return cls(
            '/attachments/%d/data.json' % attachment_id,
            singleton=True,
            session=session,
            out_type=AttachmentData,
        )