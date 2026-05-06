def find_customer(cls, session, mailbox, customer):
        """Return conversations for a specific customer in a mailbox.

        Args:
            session (requests.sessions.Session): Authenticated session.
            mailbox (helpscout.models.Mailbox): Mailbox to search.
            customer (helpscout.models.Customer): Customer to search for.

        Returns:
            RequestPaginator(output_type=helpscout.models.Conversation):
                Conversations iterator.
        """
        return cls(
            '/mailboxes/%d/customers/%s/conversations.json' % (
                mailbox.id, customer.id,
            ),
            session=session,
        )