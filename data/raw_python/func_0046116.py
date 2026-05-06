def set_asset(self, asset_id, asset_content_type=None):
        """stub"""
        if asset_id is None:
            raise NullArgument('asset_id cannot be None')
        if not isinstance(asset_id, Id):
            raise InvalidArgument('asset_id must be an instance of Id')
        if asset_content_type is not None and not isinstance(asset_content_type, Type):
            raise InvalidArgument('asset_content_type must be instance of Type')
        if asset_content_type is None:
            asset_content_type = ''
        self.my_osid_object_form._my_map['fileId'] = {
            'assetId': str(asset_id),
            'assetContentTypeId': str(asset_content_type)
        }