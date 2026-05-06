def set_assets(self, asset_ids):
        """Sets the assets.

        arg:    asset_ids (osid.id.Id[]): the asset ``Ids``
        raise:  InvalidArgument - ``asset_ids`` is invalid
        raise:  NullArgument - ``asset_ids`` is ``null``
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.ActivityForm.set_assets_template
        if not isinstance(asset_ids, list):
            raise errors.InvalidArgument()
        if self.get_assets_metadata().is_read_only():
            raise errors.NoAccess()
        idstr_list = []
        for object_id in asset_ids:
            if not self._is_valid_id(object_id):
                raise errors.InvalidArgument()
            idstr_list.append(str(object_id))
        self._my_map['assetIds'] = idstr_list