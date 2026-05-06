def assign_resource_to_bin(self, resource_id, bin_id):
        """Adds an existing ``Resource`` to a ``Bin``.

        arg:    resource_id (osid.id.Id): the ``Id`` of the ``Resource``
        arg:    bin_id (osid.id.Id): the ``Id`` of the ``Bin``
        raise:  AlreadyExists - ``resource_id`` is already assigned to
                ``bin_id``
        raise:  NotFound - ``resource_id`` or ``bin_id`` not found
        raise:  NullArgument - ``resource_id`` or ``bin_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinAssignmentSession.assign_resource_to_bin
        mgr = self._get_provider_manager('RESOURCE', local=True)
        lookup_session = mgr.get_bin_lookup_session(proxy=self._proxy)
        lookup_session.get_bin(bin_id)  # to raise NotFound
        self._assign_object_to_catalog(resource_id, bin_id)