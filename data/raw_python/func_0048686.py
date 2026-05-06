def set_avatar(self, asset_id):
        """Sets the avatar asset.

        arg:    asset_id (osid.id.Id): an asset ``Id``
        raise:  InvalidArgument - ``asset_id`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.set_avatar_template
        if self.get_avatar_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_id(asset_id):
            raise errors.InvalidArgument()
        self._my_map['avatarId'] = str(asset_id)