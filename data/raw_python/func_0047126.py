def use_plenary_book_view(self):
        """Pass through to provider CommentBookSession.use_plenary_book_view"""
        self._book_view = PLENARY
        # self._get_provider_session('comment_book_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_plenary_book_view()
            except AttributeError:
                pass