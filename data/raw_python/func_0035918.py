def create_thread(cls, session, conversation, thread, imported=False):
        """Create a conversation thread.

        Please note that threads cannot be added to conversations with 100
        threads (or more), if attempted the API will respond with HTTP 412.

        Args:
            conversation (helpscout.models.Conversation): The conversation
             that the thread is being added to.
            session (requests.sessions.Session): Authenticated session.
            thread (helpscout.models.Thread): The thread to be created.
            imported (bool, optional): The ``imported`` request parameter
             enables conversations to be created for historical purposes (i.e.
             if moving from a different platform, you can import your
             history). When ``imported`` is set to ``True``, no outgoing
             emails or notifications will be generated.

        Returns:
            helpscout.models.Conversation: Conversation including newly created
                thread.
        """
        return super(Conversations, cls).create(
            session,
            thread,
            endpoint_override='/conversations/%s.json' % conversation.id,
            imported=imported,
        )