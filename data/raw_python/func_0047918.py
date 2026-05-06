def move_asset_behind(self, asset_id, composition_id, reference_id):
        """Reorders assets in a composition by moving the specified asset behind of a reference asset.

        arg:    asset_id (osid.id.Id): ``Id`` of the ``Asset``
        arg:    composition_id (osid.id.Id): ``Id`` of the
                ``Composition``
        arg:    reference_id (osid.id.Id): ``Id`` of the reference
                ``Asset``
        raise:  NotFound - ``asset_id`` or ``reference_id``  ``not found
                in composition_id``
        raise:  NullArgument - ``asset_id, reference_id`` or
                ``composition_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization fauilure
        *compliance: mandatory -- This method must be implemented.*

        """
        self._provider_session.move_asset_behind(self, asset_id, composition_id, reference_id)