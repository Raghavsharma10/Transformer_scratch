def get_asset_id_by_label(self, label):
        """stub"""
        if self.has_file(label):
            return Id(self.my_osid_object._my_map['fileIds'][label]['assetId'])
        raise IllegalState()