def create_o3d_asset(self,
                         manip=None,
                         small_ov_set=None,
                         large_ov_set=None,
                         display_name='',
                         description=''):
        """stub"""
        if manip and not isinstance(manip, ABCDataInputStream):
            raise InvalidArgument('Manipulatable object must be an ' +
                                  'osid.transport.DataInputStream object')
        if small_ov_set and not isinstance(small_ov_set, ABCDataInputStream):
            raise InvalidArgument('Small OV Set object must be an ' +
                                  'osid.transport.DataInputStream object')
        if large_ov_set and not isinstance(large_ov_set, ABCDataInputStream):
            raise InvalidArgument('Large OV Set object must be an ' +
                                  'osid.transport.DataInputStream object')
        asset_id, asset_content_id = self.create_asset(asset_type=O3D_ASSET_TYPE,
                                                       display_name=display_name,
                                                       description=description)
        if manip is not None:
            self.add_content_to_asset(asset_id=asset_id,
                                      asset_data=manip,
                                      asset_content_type=MANIP_ASSET_CONTENT_TYPE,
                                      asset_label='3d manipulatable')
        if small_ov_set is not None:
            self.add_content_to_asset(asset_id=asset_id,
                                      asset_data=small_ov_set,
                                      asset_content_type=OV_SET_SMALL_ASSET_CONTENT_TYPE,
                                      asset_label='small orthoviewset')
        if large_ov_set is not None:
            self.add_content_to_asset(asset_id=asset_id,
                                      asset_data=large_ov_set,
                                      asset_content_type=OV_SET_LARGE_ASSET_CONTENT_TYPE,
                                      asset_label='large orthoviewset')
        return asset_id