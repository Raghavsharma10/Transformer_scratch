def set_ovs_view(self, asset_data, view_name):
        """
        view_name should be frontView, sideView, or topView
        """
        if not isinstance(asset_data, DataInputStream):
            raise InvalidArgument('view file must be an ' +
                                  'osid.transport.DataInputStream object')
        if view_name not in ['frontView', 'sideView', 'topView']:
            raise InvalidArgument('View name must be frontView, sideView, or topView.')
        self.clear_file(view_name)
        self.add_file(asset_data,
                      label=view_name,
                      asset_type=OV_ASSET_TYPE,
                      asset_content_type=OV_ASSET_CONTENT_TYPE)