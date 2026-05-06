def set_branding(self, asset_ids):
        """Sets the branding.

        arg:    asset_ids (osid.id.Id[]): the new assets
        raise:  InvalidArgument - ``asset_ids`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``asset_ids`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        if asset_ids is None:
            raise NullArgument('asset_ids cannot be None')
        if self.get_branding_metadata().is_read_only():
            raise NoAccess()
        if not isinstance(asset_ids, list):
            raise InvalidArgument('asset_ids must be a list')
        if not self.my_osid_object_form._is_valid_input(asset_ids,
                                                        self.get_branding_metadata(),
                                                        array=True):
            raise InvalidArgument()
        branding_ids = []
        for asset_id in asset_ids:
            branding_ids.append(str(asset_id))
        self.my_osid_object_form._my_map['brandingIds'] = branding_ids