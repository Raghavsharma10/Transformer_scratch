def get_asset_content_lookup_session(self, proxy):
        """Pass through to provider get_asset_content_lookup_session"""
        return getattr(sessions, 'AssetContentLookupSession')(
            provider_session=self._provider_manager.get_asset_content_lookup_session(proxy),
            authz_session=self._get_authz_session(),
            proxy=proxy)