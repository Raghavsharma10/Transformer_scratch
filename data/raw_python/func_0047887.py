def get_assets_by_ids(self, asset_ids=None):
        """Gets an ``AssetList`` corresponding to the given ``IdList``.

        In plenary mode, the returned list contains all of the assets
        specified in the ``Id`` list, in the order of the list,
        including duplicates, or an error results if an ``Id`` in the
        supplied list is not found or inaccessible. Otherwise,
        inaccessible ``Assets`` may be omitted from the list and may
        present the elements in any order including returning a unique
        set.

        arg:    asset_ids (osid.id.IdList): the list of ``Ids`` to
                retrieve
        return: (osid.repository.AssetList) - the returned ``Asset
                list``
        raise:  NotFound - an ``Id`` was not found
        raise:  NullArgument - ``asset_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        return AssetList(self._provider_session.get_assets_by_ids(asset_ids),
                         self._config_map)