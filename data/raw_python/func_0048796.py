def get_choices_files_map(self):
        """stub"""
        files_map = []
        for choice in self.get_choices():
            choice = dict(choice)
            choice['smallOrthoViewSet'] = base64.b64encode(
                self._get_asset_content(Id(choice['assetId']),
                                        OV_SET_SMALL_ASSET_CONTENT_TYPE
                                        ).get_data().read())
            choice['largeOrthoViewSet'] = base64.b64encode(
                self._get_asset_content(Id(choice['assetId']),
                                        OV_SET_LARGE_ASSET_CONTENT_TYPE
                                        ).get_data().read())
            del choice['assetId']
            files_map.append(choice)
        return files_map