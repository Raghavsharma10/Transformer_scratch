def get_asset_composition_session(self):
        """Gets the session for retrieving asset compositions.

        return: (osid.repository.AssetCompositionSession) - an
                ``AssetCompositionSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_asset_composition()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_asset_composition()`` is ``true``.*

        """
        if not self.supports_asset_composition():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssetCompositionSession(runtime=self._runtime)