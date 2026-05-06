def get_assets_by_query(self, asset_query=None):
        """Gets a list of ``Assets`` matching the given asset query.

        arg:    asset_query (osid.repository.AssetQuery): the asset
                query
        return: (osid.repository.AssetList) - the returned ``AssetList``
        raise:  NullArgument - ``asset_query`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - the ``asset_query`` is not of this service
        *compliance: mandatory -- This method must be implemented.*

        """
        return AssetList(self._provider_session.get_assets_by_query(asset_query),
                         self._config_map)