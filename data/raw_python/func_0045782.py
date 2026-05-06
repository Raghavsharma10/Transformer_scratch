def get_log_entries_by_log(self, log_id):
        """Gets the list of log entries associated with a ``Log``.

        arg:    log_id (osid.id.Id): ``Id`` of a ``Log``
        return: (osid.logging.LogEntryList) - list of related logEntry
        raise:  NotFound - ``log_id`` is not found
        raise:  NullArgument - ``log_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resources_by_bin
        mgr = self._get_provider_manager('LOGGING', local=True)
        lookup_session = mgr.get_log_entry_lookup_session_for_log(log_id, proxy=self._proxy)
        lookup_session.use_isolated_log_view()
        return lookup_session.get_log_entries()