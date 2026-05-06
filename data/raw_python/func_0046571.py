def is_child_of_family(self, id_, family_id):
        """Tests if a family is a direct child of another.

        arg:    id (osid.id.Id): an ``Id``
        arg:    family_id (osid.id.Id): the ``Id`` of a family
        return: (boolean) - ``true`` if the ``id`` is a child of
                ``family_id,``  ``false`` otherwise
        raise:  NotFound - ``family_id`` is not found
        raise:  NullArgument - ``id`` or ``family_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*
        *implementation notes*: If ``id`` not found return ``false``.

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.is_child_of_bin
        if self._catalog_session is not None:
            return self._catalog_session.is_child_of_catalog(id_=id_, catalog_id=family_id)
        return self._hierarchy_session.is_child(id_=family_id, child_id=id_)