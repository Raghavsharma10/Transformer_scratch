def use_comparative_grade_entry_view(self):
        """Pass through to provider GradeEntryLookupSession.use_comparative_grade_entry_view"""
        self._object_views['grade_entry'] = COMPARATIVE
        # self._get_provider_session('grade_entry_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_comparative_grade_entry_view()
            except AttributeError:
                pass