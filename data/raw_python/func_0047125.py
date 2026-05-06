def use_comparative_book_view(self):
        """Pass through to provider CommentBookSession.use_comparative_book_view"""
        self._book_view = COMPARATIVE
        # self._get_provider_session('comment_book_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_comparative_book_view()
            except AttributeError:
                pass