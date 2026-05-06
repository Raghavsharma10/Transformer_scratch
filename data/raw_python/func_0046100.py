def get_asset_ids(self):
        """stub"""
        asset_ids = []
        for f in self.my_osid_object._my_map['fileIds']:
            asset_ids.append(Id(self.my_osid_object._my_map['fileIds'][f]['assetId']))
        return IdList(asset_ids)