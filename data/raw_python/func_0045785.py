def get_assignable_log_ids(self, log_id):
        """Gets a list of log including and under the given log node in which any log entry can be assigned.

        arg:    log_id (osid.id.Id): the ``Id`` of the ``Log``
        return: (osid.id.IdList) - list of assignable log ``Ids``
        raise:  NullArgument - ``log_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinAssignmentSession.get_assignable_bin_ids
        # This will likely be overridden by an authorization adapter
        mgr = self._get_provider_manager('LOGGING', local=True)
        lookup_session = mgr.get_log_lookup_session(proxy=self._proxy)
        logs = lookup_session.get_logs()
        id_list = []
        for log in logs:
            id_list.append(log.get_id())
        return IdList(id_list)