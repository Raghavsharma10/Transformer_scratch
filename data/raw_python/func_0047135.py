def use_federated_book_view(self):
        """Pass through to provider CommentLookupSession.use_federated_book_view"""
        self._book_view = FEDERATED
        # self._get_provider_session('comment_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_federated_book_view()
            except AttributeError:
                pass