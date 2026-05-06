def set_manip_id(self, o3d_asset_id):
        """stub"""
        if not isinstance(o3d_asset_id, ABCId):
            raise InvalidArgument('Argument must be a valid Id')
        self.add_asset(o3d_asset_id,
                       label='manip',
                       asset_content_type=MANIP_ASSET_CONTENT_TYPE)