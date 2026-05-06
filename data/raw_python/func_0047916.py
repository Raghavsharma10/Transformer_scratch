def add_asset(self, asset_id, composition_id):
        """Appends an asset to a composition.

        arg:    asset_id (osid.id.Id): ``Id`` of the ``Asset``
        arg:    composition_id (osid.id.Id): ``Id`` of the
                ``Composition``
        raise:  AlreadyExists - ``asset_id`` already part
                ``composition_id``
        raise:  NotFound - ``asset_id`` or ``composition_id`` not found
        raise:  NullArgument - ``asset_id`` or ``composition_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization fauilure
        *compliance: mandatory -- This method must be implemented.*

        """
        self._provider_session.add_asset(self, asset_id, composition_id)