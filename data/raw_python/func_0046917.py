def get_asset_lookup_session_for_repository(self, repository_id):
        """Gets the ``OsidSession`` associated with the asset lookup service for the given repository.

        arg:    repository_id (osid.id.Id): the ``Id`` of the repository
        return: (osid.repository.AssetLookupSession) - the new
                ``AssetLookupSession``
        raise:  NotFound - ``repository_id`` not found
        raise:  NullArgument - ``repository_id`` is ``null``
        raise:  OperationFailed - ``unable to complete request``
        raise:  Unimplemented - ``supports_asset_lookup()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_asset_lookup()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_asset_lookup():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.AssetLookupSession(repository_id, runtime=self._runtime)