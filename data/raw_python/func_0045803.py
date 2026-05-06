def get_child_logs(self, log_id):
        """Gets the child logs of the given ``id``.

        arg:    log_id (osid.id.Id): the ``Id`` of the ``Log`` to query
        return: (osid.logging.LogList) - the child logs of the ``id``
        raise:  NotFound - a ``Log`` identified by ``Id is`` not found
        raise:  NullArgument - ``log_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_child_bins
        if self._catalog_session is not None:
            return self._catalog_session.get_child_catalogs(catalog_id=log_id)
        return LogLookupSession(
            self._proxy,
            self._runtime).get_logs_by_ids(
                list(self.get_child_log_ids(log_id)))