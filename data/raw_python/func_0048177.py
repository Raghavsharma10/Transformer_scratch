def use_plenary_asset_composition_view(self):
        """Pass through to provider AssetCompositionSession.use_plenary_asset_composition_view"""
        self._object_views['asset_composition'] = PLENARY
        # self._get_provider_session('asset_composition_session') # To make sure the session is tracked
        for session in self._get_provider_sessions():
            try:
                session.use_plenary_asset_composition_view()
            except AttributeError:
                pass