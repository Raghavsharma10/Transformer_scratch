def update_thread(cls, session, conversation, thread):
        """Update a thread.

        Args:
            session (requests.sessions.Session): Authenticated session.
            conversation (helpscout.models.Conversation): The conversation
                that the thread belongs to.
            thread (helpscout.models.Thread): The thread to be updated.

        Returns:
            helpscout.models.Conversation: Conversation including freshly
                updated thread.
        """
        data = thread.to_api()
        data['reload'] = True
        return cls(
            '/conversations/%s/threads/%d.json' % (
                conversation.id, thread.id,
            ),
            data=data,
            request_type=RequestPaginator.PUT,
            singleton=True,
            session=session,
        )