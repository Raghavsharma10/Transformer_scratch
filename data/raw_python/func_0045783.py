def get_log_entrie_by_log(self, log_ids):
        """Gets the list of log entries corresponding to a list of ``Log``.

        arg:    log_ids (osid.id.IdList): list of log ``Ids``
        return: (osid.logging.LogEntryList) - list of log entries
        raise:  NullArgument - ``log_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resources_by_bin
        mgr = self._get_provider_manager('LOGGING', local=True)
        lookup_session = mgr.get_log_entry_lookup_session_for_log(log_ids, proxy=self._proxy)
        lookup_session.use_isolated_log_view()
        return lookup_session.get_log_entries()