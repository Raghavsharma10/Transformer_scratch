def get_parent_log_ids(self, log_id):
        """Gets the parent ``Ids`` of the given log.

        arg:    log_id (osid.id.Id): the ``Id`` of a log
        return: (osid.id.IdList) - the parent ``Ids`` of the log
        raise:  NotFound - ``log_id`` is not found
        raise:  NullArgument - ``log_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_parent_bin_ids
        if self._catalog_session is not None:
            return self._catalog_session.get_parent_catalog_ids(catalog_id=log_id)
        return self._hierarchy_session.get_parents(id_=log_id)