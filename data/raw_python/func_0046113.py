def get_file(self):
        """stub"""
        if self.has_file_asset():
            return self._get_asset_content(
                Id(self.my_osid_object._my_map['fileId']['assetId']),
                self.my_osid_object._my_map['fileId']['assetContentTypeId']).get_data()
        raise IllegalState()