def remove_asset(self, asset_id, composition_id):
        """Removes an ``Asset`` from a ``Composition``.

        arg:    asset_id (osid.id.Id): ``Id`` of the ``Asset``
        arg:    composition_id (osid.id.Id): ``Id`` of the
                ``Composition``
        raise:  NotFound - ``asset_id``  ``not found in composition_id``
        raise:  NullArgument - ``asset_id`` or ``composition_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization fauilure
        *compliance: mandatory -- This method must be implemented.*

        """
        self._provider_session.remove_asset(self, asset_id, composition_id)