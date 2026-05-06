def has_child_gradebooks(self, gradebook_id):
        """Tests if a gradebook has any children.

        arg:    gradebook_id (osid.id.Id): the ``Id`` of a gradebook
        return: (boolean) - ``true`` if the ``gradebook_id`` has
                children, ``false`` otherwise
        raise:  NotFound - ``gradebook_id`` is not found
        raise:  NullArgument - ``gradebook_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.has_child_bins
        if self._catalog_session is not None:
            return self._catalog_session.has_child_catalogs(catalog_id=gradebook_id)
        return self._hierarchy_session.has_children(id_=gradebook_id)