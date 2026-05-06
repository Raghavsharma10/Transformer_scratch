def get_log_entry_ids_by_log(self, log_id):
        """Gets the list of ``LogEntry``  ``Ids`` associated with a ``Log``.

        arg:    log_id (osid.id.Id): ``Id`` of a ``Log``
        return: (osid.id.IdList) - list of related logEntry ``Ids``
        raise:  NotFound - ``log_id`` is not found
        raise:  NullArgument - ``log_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resource_ids_by_bin
        id_list = []
        for log_entry in self.get_log_entries_by_log(log_ids):
            id_list.append(log_entry.get_id())
        return IdList(id_list)