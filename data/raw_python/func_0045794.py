def get_root_logs(self):
        """Gets the root logs in the log hierarchy.

        A node with no parents is an orphan. While all log ``Ids`` are
        known to the hierarchy, an orphan does not appear in the
        hierarchy unless explicitly added as a root node or child of
        another node.

        return: (osid.logging.LogList) - the root logs
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method is must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_root_bins
        if self._catalog_session is not None:
            return self._catalog_session.get_root_catalogs()
        return LogLookupSession(
            self._proxy,
            self._runtime).get_logs_by_ids(list(self.get_root_log_ids()))