def get_assets_by_genus_type(self, asset_genus_type=None):
        """Gets an ``AssetList`` corresponding to the given asset genus ``Type``
        which does not include assets of types derived from the specified ``Type``.

        In plenary mode, the returned list contains all known assets or
        an error results. Otherwise, the returned list may contain only
        those assets that are accessible through this session.

        arg:    asset_genus_type (osid.type.Type): an asset genus type
        return: (osid.repository.AssetList) - the returned ``Asset
                list``
        raise:  NullArgument - ``asset_genus_type`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        return AssetList(self._provider_session.get_assets_by_genus_type(asset_genus_type),
                         self._config_map)