def get_assets_by_search(self, asset_query, asset_search):
        """Pass through to provider AssetSearchSession.get_assets_by_search"""
        # Implemented from azosid template for -
        # osid.resource.ResourceSearchSession.get_resources_by_search_template
        if not self._can('search'):
            raise PermissionDenied()
        return self._provider_session.get_assets_by_search(asset_query, asset_search)