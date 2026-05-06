def unassign_resource_from_bin(self, resource_id, bin_id):
        """Removes a ``Resource`` from a ``Bin``.

        arg:    resource_id (osid.id.Id): the ``Id`` of the ``Resource``
        arg:    bin_id (osid.id.Id): the ``Id`` of the ``Bin``
        raise:  NotFound - ``resource_id`` or ``bin_id`` not found or
                ``resource_id`` not assigned to ``bin_id``
        raise:  NullArgument - ``resource_id`` or ``bin_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinAssignmentSession.unassign_resource_from_bin
        mgr = self._get_provider_manager('RESOURCE', local=True)
        lookup_session = mgr.get_bin_lookup_session(proxy=self._proxy)
        lookup_session.get_bin(bin_id)  # to raise NotFound
        self._unassign_object_from_catalog(resource_id, bin_id)