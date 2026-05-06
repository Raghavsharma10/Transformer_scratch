def add_asset(self, asset_id, asset_content_id=None, label=None, asset_content_type=None):
        """stub"""
        if asset_id is None:
            raise NullArgument('asset_id cannot be None')
        if not isinstance(asset_id, Id):
            raise InvalidArgument('asset_id must be an Id instance')
        if asset_content_id is not None and not isinstance(asset_content_id, Id):
            raise InvalidArgument('asset_content_id must be an Id instance')
        if asset_content_type is not None and not isinstance(asset_content_type, Type):
            raise InvalidArgument('asset_content_type must be a Type instance')
        if label is None:
            label = self._label_metadata['default_string_values'][0]
        else:
            if not self.my_osid_object_form._is_valid_string(
                    label, self.get_label_metadata()) or '.' in label:
                raise InvalidArgument('label')
        if asset_content_type is None:
            asset_content_type = ''

        self.my_osid_object_form._my_map['fileIds'][label] = {
            'assetId': str(asset_id),
            'assetContentTypeId': str(asset_content_type)
        }

        if asset_content_id is not None:
            self.my_osid_object_form._my_map['fileIds'][label].update({
                'assetContentId': str(asset_content_id)
            })