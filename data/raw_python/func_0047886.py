def get_asset(self, asset_id=None):
        """Gets the ``Asset`` specified by its ``Id``.

        In plenary mode, the exact ``Id`` is found or a ``NotFound``
        results. Otherwise, the returned ``Asset`` may have a different
        ``Id`` than requested, such as the case where a duplicate ``Id``
        was assigned to an ``Asset`` and retained for compatibility.

        arg:    asset_id (osid.id.Id): the ``Id`` of the ``Asset`` to
                retrieve
        return: (osid.repository.Asset) - the returned ``Asset``
        raise:  NotFound - no ``Asset`` found with the given ``Id``
        raise:  NullArgument - ``asset_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        return Asset(self._provider_session.get_asset(asset_id), self._config_map)