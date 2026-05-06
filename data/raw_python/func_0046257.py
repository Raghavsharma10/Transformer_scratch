def set_provider_links(self, resource_ids=None):
        """Sets a provider chain in order from the most recent source to
        the originating source.

        :param resource_ids: the new source
        :type resource_ids: ``osid.id.Id[]``
        :raise: ``InvalidArgument`` -- ``resource_ids`` is invalid
        :raise: ``NoAccess`` -- ``Metadata.isReadOnly()`` is ``true``
        :raise: ``NullArgument`` -- ``resource_ids`` is ``null``

        *compliance: mandatory -- This method must be implemented.*

        """
        if resource_ids is None:
            raise NullArgument()
        metadata = Metadata(**settings.METADATA['provider_link_ids'])
        if metadata.is_read_only():
            raise NoAccess()
        if self._is_valid_input(resource_ids, metadata, array=True):
            self._my_map['providerLinkIds'] = []
            for i in resource_ids:
                self._my_map['providerLinkIds'].append(str(i))
        else:
            raise InvalidArgument()