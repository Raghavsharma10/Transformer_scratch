def is_ancestor_of_log(self, id_, log_id):
        """Tests if an ``Id`` is an ancestor of a log.

        arg:    id (osid.id.Id): an ``Id``
        arg:    log_id (osid.id.Id): the ``Id`` of a log
        return: (boolean) - ``true`` if the ``id`` is an ancestor of the
                ``log_id,``  ``false`` otherwise
        raise:  NotFound - ``log_id`` is not found
        raise:  NullArgument - ``id`` or ``log_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*
        *implementation notes*: If ``id`` is not found return ``false``.

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.is_ancestor_of_bin
        if self._catalog_session is not None:
            return self._catalog_session.is_ancestor_of_catalog(id_=id_, catalog_id=log_id)
        return self._hierarchy_session.is_ancestor(id_=id_, ancestor_id=log_id)