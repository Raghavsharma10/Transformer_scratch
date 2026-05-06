def remove_child_logs(self, log_id):
        """Removes all children from a log.

        arg:    log_id (osid.id.Id): the ``Id`` of a log
        raise:  NotFound - ``log_id`` not in hierarchy
        raise:  NullArgument - ``log_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchyDesignSession.remove_child_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.remove_child_catalogs(catalog_id=log_id)
        return self._hierarchy_session.remove_children(id_=log_id)