def get_asset_lookup_session(self):
        """Gets the ``OsidSession`` associated with the asset lookup service.

        return: (osid.repository.AssetLookupSession) - the new
                ``AssetLookupSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_asset_lookup()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_asset_lookup()`` is ``true``.*

        """
        if not self.supports_asset_lookup():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssetLookupSession(runtime=self._runtime)