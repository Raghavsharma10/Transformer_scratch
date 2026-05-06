def get_source_id(self):
        """Gets the ``Resource Id`` of the source of this asset.

        The source is the original owner of the copyright of this asset
        and may differ from the creator of this asset. The source for a
        published book written by Margaret Mitchell would be Macmillan.
        The source for an unpublished painting by Arthur Goodwin would
        be Arthur Goodwin.

        An ``Asset`` is ``Sourceable`` and also contains a provider
        identity. The provider is the entity that makes this digital
        asset available in this repository but may or may not be the
        publisher of the contents depicted in the asset. For example, a
        map published by Ticknor and Fields in 1848 may have a provider
        of Library of Congress and a source of Ticknor and Fields. If
        copied from a repository at Middlebury College, the provider
        would be Middlebury College and a source of Ticknor and Fields.

        return: (osid.id.Id) - the source ``Id``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.Resource.get_avatar_id_template
        if not bool(self._my_map['sourceId']):
            raise errors.IllegalState('this Asset has no source')
        else:
            return Id(self._my_map['sourceId'])