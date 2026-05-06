def get_asset_repository_assignment_session(self):
        """Gets the session for assigning asset to repository mappings.

        return: (osid.repository.AssetRepositoryAssignmentSession) - an
                ``AssetRepositoryAsignmentSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented -
                ``supports_asset_repository_assignment()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_asset_repository_assignment()`` is ``true``.*

        """
        if not self.supports_asset_repository_assignment():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssetRepositoryAssignmentSession(runtime=self._runtime)