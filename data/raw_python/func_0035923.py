def list(cls, session, mailbox):
        """Return conversations in a mailbox.

        Args:
            session (requests.sessions.Session): Authenticated session.
            mailbox (helpscout.models.Mailbox): Mailbox to list.

        Returns:
            RequestPaginator(output_type=helpscout.models.Conversation):
                Conversations iterator.
        """
        endpoint = '/mailboxes/%d/conversations.json' % mailbox.id
        return super(Conversations, cls).list(session, endpoint)