def use_comparative_composition_repository_view(self):
        """Pass through to provider CompositionRepositorySession.use_comparative_composition_repository_view"""
        self._repository_view = COMPARATIVE
        # self._get_provider_session('composition_repository_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_comparative_repository_view()
            except AttributeError:
                pass