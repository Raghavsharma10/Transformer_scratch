def add_root_log(self, log_id):
        """Adds a root log.

        arg:    log_id (osid.id.Id): the ``Id`` of a log
        raise:  AlreadyExists - ``log_id`` is already in hierarchy
        raise:  NotFound - ``log_id`` is not found
        raise:  NullArgument - ``log_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchyDesignSession.add_root_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.add_root_catalog(catalog_id=log_id)
        return self._hierarchy_session.add_root(id_=log_id)