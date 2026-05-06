def unassign_item_from_bank(self, item_id, bank_id):
        """Removes an ``Item`` from a ``Bank``.

        arg:    item_id (osid.id.Id): the ``Id`` of the ``Item``
        arg:    bank_id (osid.id.Id): the ``Id`` of the ``Bank``
        raise:  NotFound - ``item_id`` or ``bank_id`` not found or
                ``item_id`` not assigned to ``bank_id``
        raise:  NullArgument - ``item_id`` or ``bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinAssignmentSession.unassign_resource_from_bin
        mgr = self._get_provider_manager('ASSESSMENT', local=True)
        lookup_session = mgr.get_bank_lookup_session(proxy=self._proxy)
        lookup_session.get_bank(bank_id)  # to raise NotFound
        self._unassign_object_from_catalog(item_id, bank_id)