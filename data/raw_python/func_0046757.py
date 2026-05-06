def get_parent_gradebook_ids(self, gradebook_id):
        """Gets the parent ``Ids`` of the given gradebook.

        arg:    gradebook_id (osid.id.Id): the ``Id`` of a gradebook
        return: (osid.id.IdList) - the parent ``Ids`` of the gradebook
        raise:  NotFound - ``gradebook_id`` is not found
        raise:  NullArgument - ``gradebook_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_parent_bin_ids
        if self._catalog_session is not None:
            return self._catalog_session.get_parent_catalog_ids(catalog_id=gradebook_id)
        return self._hierarchy_session.get_parents(id_=gradebook_id)