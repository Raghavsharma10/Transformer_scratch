def get_child_bin_ids(self, bin_id):
        """Gets the child ``Ids`` of the given bin.

        arg:    bin_id (osid.id.Id): the ``Id`` to query
        return: (osid.id.IdList) - the children of the bin
        raise:  NotFound - ``bin_id`` not found
        raise:  NullArgument - ``bin_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_child_bin_ids
        if self._catalog_session is not None:
            return self._catalog_session.get_child_catalog_ids(catalog_id=bin_id)
        return self._hierarchy_session.get_children(id_=bin_id)