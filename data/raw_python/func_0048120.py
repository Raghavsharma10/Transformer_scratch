def use_comparative_family_view(self):
        """Pass through to provider FamilyLookupSession.use_comparative_family_view"""
        self._family_view = COMPARATIVE
        # self._get_provider_session('family_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_comparative_family_view()
            except AttributeError:
                pass