def remove_root_log(self, log_id):
        """Removes a root log.

        arg:    log_id (osid.id.Id): the ``Id`` of a log
        raise:  NotFound - ``log_id`` is not a root
        raise:  NullArgument - ``log_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchyDesignSession.remove_root_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.remove_root_catalog(catalog_id=log_id)
        return self._hierarchy_session.remove_root(id_=log_id)