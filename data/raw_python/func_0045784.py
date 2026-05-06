def get_log_ids_by_log_entry(self, log_entry_id):
        """Gets the list of ``Log``  ``Ids`` mapped to a ``LogEntry``.

        arg:    log_entry_id (osid.id.Id): ``Id`` of a ``LogEntry``
        return: (osid.id.IdList) - list of log ``Ids``
        raise:  NotFound - ``log_entry_id`` is not found
        raise:  NullArgument - ``log_entry_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_bin_ids_by_resource
        mgr = self._get_provider_manager('LOGGING', local=True)
        lookup_session = mgr.get_log_entry_lookup_session(proxy=self._proxy)
        lookup_session.use_federated_log_view()
        log_entry = lookup_session.get_log_entry(log_entry_id)
        id_list = []
        for idstr in log_entry._my_map['assignedLogIds']:
            id_list.append(Id(idstr))
        return IdList(id_list)