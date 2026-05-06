def set_provider_links(self, resource_ids):
        """Sets a provider chain in order from the most recent source to the originating source.

        arg:    resource_ids (osid.id.Id[]): the new source
        raise:  InvalidArgument - ``resource_ids`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``resource_ids`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.ActivityForm.set_assets_template
        if not isinstance(resource_ids, list):
            raise errors.InvalidArgument()
        if self.get_provider_links_metadata().is_read_only():
            raise errors.NoAccess()
        idstr_list = []
        for object_id in resource_ids:
            if not self._is_valid_id(object_id):
                raise errors.InvalidArgument()
            idstr_list.append(str(object_id))
        self._my_map['providerLinkIds'] = idstr_list