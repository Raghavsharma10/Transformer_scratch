def create(cls, session, record, imported=False, auto_reply=False):
        """Create a conversation.

        Please note that conversation cannot be created with more than 100
        threads, if attempted the API will respond with HTTP 412.

        Args:
            session (requests.sessions.Session): Authenticated session.
            record (helpscout.models.Conversation): The conversation
             to be created.
            imported (bool, optional): The ``imported`` request parameter
             enables conversations to be created for historical purposes (i.e.
             if moving from a different platform, you can import your
             history). When ``imported`` is set to ``True``, no outgoing
             emails or notifications will be generated.
            auto_reply (bool): The ``auto_reply`` request parameter enables
             auto replies to be sent when a conversation is created via the
             API. When ``auto_reply`` is set to ``True``, an auto reply will
             be sent as long as there is at least one ``customer`` thread in
             the conversation.

        Returns:
            helpscout.models.Conversation: Newly created conversation.
        """
        return super(Conversations, cls).create(
            session,
            record,
            imported=imported,
            auto_reply=auto_reply,
        )