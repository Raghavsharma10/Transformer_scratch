def remove_child_gradebook(self, gradebook_id, child_id):
        """Removes a child from a gradebook.

        arg:    gradebook_id (osid.id.Id): the ``Id`` of a gradebook
        arg:    child_id (osid.id.Id): the ``Id`` of the new child
        raise:  NotFound - ``gradebook_id`` not a parent of ``child_id``
        raise:  NullArgument - ``gradebook_id`` or ``child_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchyDesignSession.remove_child_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.remove_child_catalog(catalog_id=gradebook_id, child_id=child_id)
        return self._hierarchy_session.remove_child(id_=gradebook_id, child_id=child_id)