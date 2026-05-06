def set_ortho_choice(self, small_asset_data, large_asset_data, name='Choice'):
        """stub"""
        o3d_asset_id = self.create_o3d_asset(manip=None,
                                             small_ov_set=small_asset_data,
                                             large_ov_set=large_asset_data,
                                             display_name=name)
        self.add_choice(o3d_asset_id, name=name)