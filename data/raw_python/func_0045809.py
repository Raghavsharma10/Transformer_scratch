def remove_child_log(self, log_id, child_id):
        """Removes a child from a log.

        arg:    log_id (osid.id.Id): the ``Id`` of a log
        arg:    child_id (osid.id.Id): the ``Id`` of the new child
        raise:  NotFound - ``log_id`` not a parent of ``child_id``
        raise:  NullArgument - ``log_id`` or ``child_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchyDesignSession.remove_child_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.remove_child_catalog(catalog_id=log_id, child_id=child_id)
        return self._hierarchy_session.remove_child(id_=log_id, child_id=child_id)