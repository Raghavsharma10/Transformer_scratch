def has_file_url(self):
        """stub"""
        return bool(self._get_asset_content(
            Id(self.my_osid_object._my_map['fileId']['assetId']),
            self.my_osid_object._my_map['fileId']['assetContentTypeId']).has_url())