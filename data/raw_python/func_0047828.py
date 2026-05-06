def use_comparative_proficiency_view(self):
        """Pass through to provider ProficiencyLookupSession.use_comparative_proficiency_view"""
        self._object_views['proficiency'] = COMPARATIVE
        # self._get_provider_session('proficiency_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_comparative_proficiency_view()
            except AttributeError:
                pass