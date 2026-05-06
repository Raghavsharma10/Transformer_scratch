def add_file(self,
                 asset_data,
                 label=None,
                 asset_type=None,
                 asset_content_type=None,
                 asset_content_record_types=None,
                 asset_name='',
                 asset_description=''):
        """stub"""
        if asset_data is None:
            raise NullArgument('asset_data cannot be None')
        if not isinstance(asset_data, DataInputStream):
            raise InvalidArgument('asset_data must be instance of DataInputStream')
        if asset_type is not None and not isinstance(asset_type, Type):
            raise InvalidArgument('asset_type must be an instance of Type')
        if asset_content_type is not None and not isinstance(asset_content_type, Type):
            raise InvalidArgument('asset_content_type must be an instance of Type')
        if asset_content_record_types is not None and not isinstance(asset_content_record_types, list):
            raise InvalidArgument('asset_content_record_types must be an instance of list')
        if asset_content_record_types is not None:
            for record_type in asset_content_record_types:
                if not isinstance(record_type, Type):
                    raise InvalidArgument('non-Type present in asset_content_record_types')

        if label is None:
            label = self._label_metadata['default_string_values'][0]
        else:
            if not self.my_osid_object_form._is_valid_string(
                    label, self.get_label_metadata()) or '.' in label:
                raise InvalidArgument('label')

        asset_id, asset_content_id = self.create_asset(asset_data=asset_data,
                                                       asset_type=asset_type,
                                                       asset_content_type=asset_content_type,
                                                       asset_content_record_types=asset_content_record_types,
                                                       display_name=asset_name,
                                                       description=asset_description)
        self.add_asset(asset_id,
                       asset_content_id,
                       label,
                       asset_content_type)