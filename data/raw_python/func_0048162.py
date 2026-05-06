def use_plenary_repository_view(self):
        """Pass through to provider AssetRepositorySession.use_plenary_repository_view"""
        self._repository_view = PLENARY
        # self._get_provider_session('asset_repository_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_plenary_repository_view()
            except AttributeError:
                pass