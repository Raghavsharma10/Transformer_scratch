def get_folders(cls, session, mailbox_or_id):
        """List the folders for the mailbox.

        Args:
            mailbox_or_id (helpscout.models.Mailbox or int): Mailbox or the ID
                of the mailbox to get the folders for.

        Returns:
            RequestPaginator(output_type=helpscout.models.Folder): Folders
                iterator.
        """
        if isinstance(mailbox_or_id, Mailbox):
            mailbox_or_id = mailbox_or_id.id
        return cls(
            '/mailboxes/%d/folders.json' % mailbox_or_id,
            session=session,
            out_type=Folder,
        )