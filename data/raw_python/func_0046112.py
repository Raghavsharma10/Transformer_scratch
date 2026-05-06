def get_file_url(self):
        """stub"""
        if self.has_file_url():
            return self._get_asset_content(
                Id(self.my_osid_object._my_map['fileId']['assetId']),
                self.my_osid_object._my_map['fileId']['assetContentTypeId']).get_url()
        raise IllegalState()