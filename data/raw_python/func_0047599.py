def get_object_map(self):
        """stub"""
        obj_map = self._payload.get_object_map()
        obj_map['assetContents'] = []
        for asset_content in self.get_asset_contents():
            obj_map['assetContents'].append(asset_content.object_map)
        return obj_map