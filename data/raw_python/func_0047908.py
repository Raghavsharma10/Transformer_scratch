def _get_asset_content(self, asset_content_id):
        """stub"""
        asset_content = None
        for asset in self._asset_lookup_session.get_assets():
            for content in asset.get_asset_contents():
                if content.get_id() == asset_content_id:
                    asset_content = content
                    break
            if asset_content is not None:
                break
        if asset_content is None:
            raise NotFound('THe AWS Adapter could not find AssetContent ' +
                           str(asset_content_id))
        return asset_content