def get_assets_by_provider(self, resource_id=None):
        """Gets an ``AssetList`` from the given provider.

        In plenary mode, the returned list contains all known assets or
        an error results. Otherwise, the returned list may contain only
        those assets that are accessible through this session.

        arg:    resource_id (osid.id.Id): a resource ``Id``
        return: (osid.repository.AssetList) - the returned ``Asset
                list``
        raise:  NullArgument - ``resource_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        return AssetList(self._provider_session.get_assets_by_provider(resource_id),
                         self._config_map)