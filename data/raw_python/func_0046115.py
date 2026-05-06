def set_file(self,
                 asset_data=None,
                 asset_type=None,
                 asset_content_type=None,
                 asset_name='',
                 asset_description=''):
        """stub"""
        if asset_data is None:
            raise NullArgument()
        if not isinstance(asset_data, DataInputStream):
            raise InvalidArgument('asset_data must be instance of DataInputStream')
        if asset_type is not None and not isinstance(asset_type, Type):
            raise InvalidArgument('asset_type must be instance of Type')
        if asset_content_type is not None and not isinstance(asset_content_type, Type):
            raise InvalidArgument('asset_content_type must be instance of Type')

        asset_id, asset_content_id = self.create_asset(asset_data=asset_data,
                                                       asset_type=asset_type,
                                                       asset_content_type=asset_content_type,
                                                       display_name=asset_name,
                                                       description=asset_description)
        self.set_asset(asset_id,
                       asset_content_type)