def add_root_gradebook(self, gradebook_id):
        """Adds a root gradebook.

        arg:    gradebook_id (osid.id.Id): the ``Id`` of a gradebook
        raise:  AlreadyExists - ``gradebook_id`` is already in hierarchy
        raise:  NotFound - ``gradebook_id`` not found
        raise:  NullArgument - ``gradebook_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchyDesignSession.add_root_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.add_root_catalog(catalog_id=gradebook_id)
        return self._hierarchy_session.add_root(id_=gradebook_id)