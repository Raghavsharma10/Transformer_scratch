def find_user(cls, session, mailbox, user):
        """Return conversations for a specific user in a mailbox.

        Args:
            session (requests.sessions.Session): Authenticated session.
            mailbox (helpscout.models.Mailbox): Mailbox to search.
            user (helpscout.models.User): User to search for.

        Returns:
            RequestPaginator(output_type=helpscout.models.Conversation):
                Conversations iterator.
        """
        return cls(
            '/mailboxes/%d/users/%s/conversations.json' % (
                mailbox.id, user.id,
            ),
            session=session,
        )