def is_parent_of_gradebook(self, id_, gradebook_id):
        """Tests if an ``Id`` is a direct parent of a gradebook.

        arg:    id (osid.id.Id): an ``Id``
        arg:    gradebook_id (osid.id.Id): the ``Id`` of a gradebook
        return: (boolean) - ``true`` if this ``id`` is a parent of
                ``gradebook_id,``  ``false`` otherwise
        raise:  NotFound - ``gradebook_id`` is not found
        raise:  NullArgument - ``id`` or ``gradebook_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*
        *implementation notes*: If ``id`` not found return ``false``.

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.is_parent_of_bin
        if self._catalog_session is not None:
            return self._catalog_session.is_parent_of_catalog(id_=id_, catalog_id=gradebook_id)
        return self._hierarchy_session.is_parent(id_=gradebook_id, parent_id=id_)