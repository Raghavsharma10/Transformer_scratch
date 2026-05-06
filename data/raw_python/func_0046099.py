def get_file_urls_map(self):
        """stub"""
        file_urls_map = {}
        if self.has_files():
            for label in self.my_osid_object._my_map['fileIds']:
                label_map = self.my_osid_object._my_map['fileIds'][label]
                if 'assetContentId' in label_map and bool(label_map['assetContentId']):
                    asset_content = self._get_asset_content(
                        Id(label_map['assetId']),
                        asset_content_id=Id(label_map['assetContentId']))
                else:
                    asset_content = self._get_asset_content(
                        Id(label_map['assetId']),
                        asset_content_type_str=label_map['assetContentTypeId'])
                file_urls_map[label] = asset_content.get_url()
            return file_urls_map
        raise IllegalState('no files_map')