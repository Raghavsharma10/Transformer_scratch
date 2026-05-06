def add_root_bin(self, bin_id):
        """Adds a root bin.

        arg:    bin_id (osid.id.Id): the ``Id`` of a bin
        raise:  AlreadyExists - ``bin_id`` is already in hierarchy
        raise:  NotFound - ``bin_id`` not found
        raise:  NullArgument - ``bin_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchyDesignSession.add_root_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.add_root_catalog(catalog_id=bin_id)
        return self._hierarchy_session.add_root(id_=bin_id)