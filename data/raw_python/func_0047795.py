def remove_child_bin(self, bin_id, child_id):
        """Removes a child from a bin.

        arg:    bin_id (osid.id.Id): the ``Id`` of a bin
        arg:    child_id (osid.id.Id): the ``Id`` of the new child
        raise:  NotFound - ``bin_id`` not a parent of ``child_id``
        raise:  NullArgument - ``bin_id`` or ``child_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchyDesignSession.remove_child_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.remove_child_catalog(catalog_id=bin_id, child_id=child_id)
        return self._hierarchy_session.remove_child(id_=bin_id, child_id=child_id)