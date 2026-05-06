def get_asset_content_lookup_session_for_repository(self, repository_id):
        """Pass through to provider get_asset_content_lookup_session_for_repository"""
        return getattr(sessions, 'AssetContentLookupSession')(
            provider_session=self._provider_manager.get_asset_content_lookup_session_for_repository(repository_id),
            authz_session=self._authz_session)