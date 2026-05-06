def use_plenary_asset_view(self):
        """Pass through to provider AssetLookupSession.use_plenary_asset_view"""
        self._object_views['asset'] = PLENARY
        # self._get_provider_session('asset_lookup_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_plenary_asset_view()
            except AttributeError:
                pass