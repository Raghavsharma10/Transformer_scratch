def use_comparative_gradebook_column_view(self):
        """Pass through to provider GradebookColumnLookupSession.use_comparative_gradebook_column_view"""
        self._object_views['gradebook_column'] = COMPARATIVE
        # self._get_provider_session('gradebook_column_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_comparative_gradebook_column_view()
            except AttributeError:
                pass