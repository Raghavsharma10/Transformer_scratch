def has_child_bins(self, bin_id):
        """Tests if a bin has any children.

        arg:    bin_id (osid.id.Id): the ``Id`` of a bin
        return: (boolean) - ``true`` if the ``bin_id`` has children,
                ``false`` otherwise
        raise:  NotFound - ``bin_id`` not found
        raise:  NullArgument - ``bin_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.has_child_bins
        if self._catalog_session is not None:
            return self._catalog_session.has_child_catalogs(catalog_id=bin_id)
        return self._hierarchy_session.has_children(id_=bin_id)