def find_in_mailbox(cls, session, mailbox_or_id):
        """Get the users that are associated to a Mailbox.

        Args:
            session (requests.sessions.Session): Authenticated session.
            mailbox_or_id (MailboxRef or int): Mailbox of the ID of the
                mailbox to get the folders for.

        Returns:
            RequestPaginator(output_type=helpscout.models.User): Users
                iterator.
        """
        if hasattr(mailbox_or_id, 'id'):
            mailbox_or_id = mailbox_or_id.id
        return cls(
            '/mailboxes/%d/users.json' % mailbox_or_id,
            session=session,
        )