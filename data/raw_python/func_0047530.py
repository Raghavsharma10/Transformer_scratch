def set_ortho_view_set(self, front_view, side_view, top_view):
        """stub"""
        if (not isinstance(front_view, DataInputStream) or
                not isinstance(top_view, DataInputStream) or
                not isinstance(side_view, DataInputStream)):
            raise InvalidArgument('views must be osid.transport.DataInputStream objects')
        self.add_file(front_view,
                      label='frontView',
                      asset_type=OV_ASSET_TYPE,
                      asset_content_type=OV_ASSET_CONTENT_TYPE)
        self.add_file(side_view,
                      label='sideView',
                      asset_type=OV_ASSET_TYPE,
                      asset_content_type=OV_ASSET_CONTENT_TYPE)
        self.add_file(top_view,
                      label='topView',
                      asset_type=OV_ASSET_TYPE,
                      asset_content_type=OV_ASSET_CONTENT_TYPE)