def get_asset_repository_session(self):
        """Gets the session for retrieving asset to repository mappings.

        return: (osid.repository.AssetRepositorySession) - an
                ``AssetRepositorySession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_asset_repository()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_asset_repository()`` is ``true``.*

        """
        if not self.supports_asset_repository():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssetRepositorySession(runtime=self._runtime)