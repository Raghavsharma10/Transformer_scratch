def set_provider(self, provider_id):
        """Sets a provider.

        arg:    provider_id (osid.id.Id): the new provider
        raise:  InvalidArgument - ``provider_id`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``provider_id`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        if provider_id is None:
            raise NullArgument('provider_id cannot be None')
        if self.get_provider_metadata().is_read_only():
            raise NoAccess()
        if not self.my_osid_object_form._is_valid_id(provider_id):
            raise InvalidArgument('provider_id must be instance of Id')
        self.my_osid_object_form._my_map['providerId'] = str(provider_id)