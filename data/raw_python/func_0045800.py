def has_child_logs(self, log_id):
        """Tests if a log has any children.

        arg:    log_id (osid.id.Id): the ``Id`` of a log
        return: (boolean) - ``true`` if the ``log_id`` has children,
                ``false`` otherwise
        raise:  NotFound - ``log_id`` is not found
        raise:  NullArgument - ``log_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.has_child_bins
        if self._catalog_session is not None:
            return self._catalog_session.has_child_catalogs(catalog_id=log_id)
        return self._hierarchy_session.has_children(id_=log_id)