def get_asset_query_session(self):
        """Gets an asset query session.

        return: (osid.repository.AssetQuerySession) - an
                ``AssetQuerySession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_asset_query()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_asset_query()`` is ``true``.*

        """
        if not self.supports_asset_query():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssetQuerySession(runtime=self._runtime)