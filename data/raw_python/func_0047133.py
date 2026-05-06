def use_comparative_comment_view(self):
        """Pass through to provider CommentLookupSession.use_comparative_comment_view"""
        self._object_views['comment'] = COMPARATIVE
        # self._get_provider_session('comment_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_comparative_comment_view()
            except AttributeError:
                pass