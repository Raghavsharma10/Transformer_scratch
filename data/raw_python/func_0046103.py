def get_file_by_label(self, label, asset_content_type=None):
        """stub"""
        return self._get_asset_content(self.get_asset_id_by_label(label), asset_content_type).get_data()