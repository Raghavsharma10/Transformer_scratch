def unassign_log_entry_from_log(self, log_entry_id, log_id):
        """Removes a ``LogEntry`` from a ``Log``.

        arg:    log_entry_id (osid.id.Id): the ``Id`` of the
                ``LogEntry``
        arg:    log_id (osid.id.Id): the ``Id`` of the ``Log``
        raise:  NotFound - ``log_entry_id`` or ``log_id`` not found or
                ``log_entry_id`` not assigned to ``log_id``
        raise:  NullArgument - ``log_entry_id`` or ``log_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinAssignmentSession.unassign_resource_from_bin
        mgr = self._get_provider_manager('LOGGING', local=True)
        lookup_session = mgr.get_log_lookup_session(proxy=self._proxy)
        lookup_session.get_log(log_id)  # to raise NotFound
        self._unassign_object_from_catalog(log_entry_id, log_id)