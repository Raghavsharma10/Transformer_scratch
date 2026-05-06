def get_assets_by_record_type(self, asset_record_type=None):
        """Gets an ``AssetList`` containing the given asset record ``Type``.

        In plenary mode, the returned list contains all known assets or
        an error results. Otherwise, the returned list may contain only
        those assets that are accessible through this session.

        arg:    asset_record_type (osid.type.Type): an asset record type
        return: (osid.repository.AssetList) - the returned ``Asset
                list``
        raise:  NullArgument - ``asset_record_type`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        return AssetList(self._provider_session.get_assets_by_record_type(asset_record_type),
                         self._config_map)