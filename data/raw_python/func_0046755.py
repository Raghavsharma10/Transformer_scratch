def has_parent_gradebooks(self, gradebook_id):
        """Tests if the ``Gradebook`` has any parents.

        arg:    gradebook_id (osid.id.Id): the ``Id`` of a gradebook
        return: (boolean) - ``true`` if the gradebook has parents,
                ``false`` otherwise
        raise:  NotFound - ``gradebook_id`` is not found
        raise:  NullArgument - ``gradebook_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.has_parent_bins
        if self._catalog_session is not None:
            return self._catalog_session.has_parent_catalogs(catalog_id=gradebook_id)
        return self._hierarchy_session.has_parents(id_=gradebook_id)