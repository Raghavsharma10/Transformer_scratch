def get_parent_logs(self, log_id):
        """Gets the parent logs of the given ``id``.

        arg:    log_id (osid.id.Id): the ``Id`` of the ``Log`` to query
        return: (osid.logging.LogList) - the parent logs of the ``id``
        raise:  NotFound - a ``Log`` identified by ``Id is`` not found
        raise:  NullArgument - ``log_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_parent_bins
        if self._catalog_session is not None:
            return self._catalog_session.get_parent_catalogs(catalog_id=log_id)
        return LogLookupSession(
            self._proxy,
            self._runtime).get_logs_by_ids(
                list(self.get_parent_log_ids(log_id)))