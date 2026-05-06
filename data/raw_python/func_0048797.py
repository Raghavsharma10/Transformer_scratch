def get_choices_file_urls_map(self):
        """stub"""
        file_urls_map = []
        for choice in self.get_choices():
            choice = dict(choice)
            small_asset_content = self._get_asset_content(
                Id(choice['assetId']), OV_SET_SMALL_ASSET_CONTENT_TYPE)
            choice['smallOrthoViewSet'] = small_asset_content.get_url()

            small_asset_content = self._get_asset_content(
                Id(choice['assetId']), OV_SET_LARGE_ASSET_CONTENT_TYPE)
            choice['largeOrthoViewSet'] = small_asset_content.get_url()

            del choice['assetId']
            file_urls_map.append(choice)
        return file_urls_map