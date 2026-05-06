def get_files_map(self):
        """stub"""
        files_map = {}
        if self.has_files():
            for label in self.my_osid_object._my_map['fileIds']:
                asset_content = self._get_asset_content(
                    Id(self.my_osid_object._my_map['fileIds'][label]['assetId']),
                    Type(self.my_osid_object._my_map['fileIds'][label]['assetContentTypeId']))
                try:
                    files_map[label] = asset_content._my_map['base64']
                except KeyError:
                    files_map[label] = base64.b64encode(asset_content.get_data().read())
            return files_map
        raise IllegalState('no files_map')