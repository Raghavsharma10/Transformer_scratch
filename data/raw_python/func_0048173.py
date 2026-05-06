def use_isolated_repository_view(self):
        """Pass through to provider AssetLookupSession.use_isolated_repository_view"""
        self._repository_view = ISOLATED
        # self._get_provider_session('asset_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_isolated_repository_view()
            except AttributeError:
                pass