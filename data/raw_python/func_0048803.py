def set_manip(self, manip, ovs_sm=None, ovs_lg=None, name='A manipulatable'):
        """stub"""
        o3d_manip_id = self.create_o3d_asset(manip,
                                             small_ov_set=ovs_sm,
                                             large_ov_set=ovs_lg,
                                             display_name=name)
        self.set_manip_id(o3d_manip_id)