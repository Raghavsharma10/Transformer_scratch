def add_content_to_asset(self,
                             asset_id,
                             asset_data=None,
                             asset_url=None,
                             asset_content_type=None,
                             asset_label=None):
        """stub"""
        # This method creates a new AssetContent related to the given asset_id:
        return self._add_asset_content(asset_id=asset_id,
                                       asset_data=asset_data,
                                       asset_url=asset_url,
                                       asset_content_type=asset_content_type,
                                       asset_label=asset_label)