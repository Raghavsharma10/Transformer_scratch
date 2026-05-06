def get_asset_ids_map(self):
        """stub"""
        asset_ids_map = {}
        for label, asset_obj in self.my_osid_object._my_map['fileIds'].items():
            asset_ids_map[label] = asset_obj
        return asset_ids_map