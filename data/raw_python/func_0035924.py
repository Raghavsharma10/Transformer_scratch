def list_folder(cls, session, mailbox, folder):
        """Return conversations in a specific folder of a mailbox.

        Args:
            session (requests.sessions.Session): Authenticated session.
            mailbox (helpscout.models.Mailbox): Mailbox that folder is in.
            folder (helpscout.models.Folder): Folder to list.

        Returns:
            RequestPaginator(output_type=helpscout.models.Conversation):
                Conversations iterator.
        """
        return cls(
            '/mailboxes/%d/folders/%s/conversations.json' % (
                mailbox.id, folder.id,
            ),
            session=session,
        )