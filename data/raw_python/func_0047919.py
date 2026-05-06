def order_assets(self, asset_ids, composition_id):
        """Reorders a set of assets in a composition.

        arg:    asset_ids (osid.id.Id[]): ``Ids`` for a set of
                ``Assets``
        arg:    composition_id (osid.id.Id): ``Id`` of the
                ``Composition``
        raise:  NotFound - ``composition_id`` not found or, an
                ``asset_id`` not related to ``composition_id``
        raise:  NullArgument - ``instruction_ids`` or ``agenda_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        self._provider_session.order_assets(self, asset_ids, composition_id)