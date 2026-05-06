def remove_child_bins(self, bin_id):
        """Removes all children from a bin.

        arg:    bin_id (osid.id.Id): the ``Id`` of a bin
        raise:  NotFound - ``bin_id`` not in hierarchy
        raise:  NullArgument - ``bin_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchyDesignSession.remove_child_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.remove_child_catalogs(catalog_id=bin_id)
        return self._hierarchy_session.remove_children(id_=bin_id)