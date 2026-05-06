def is_child(self, id_, child_id):
        """Tests if a node is a direct child of another.

        arg:    id (osid.id.Id): the ``Id`` to query
        arg:    child_id (osid.id.Id): the ``Id`` of a child
        return: (boolean) - ``true`` if this ``child_id`` is a child of
                the ``Id,``  ``false`` otherwise
        raise:  NotFound - ``id`` is not found
        raise:  NullArgument - ``id`` or ``child_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*
        *implementation notes*: If ``child_id`` not found return
        ``false``.

        """
        return bool(self._rls.get_relationships_by_genus_type_for_peers(
            id_,
            child_id,
            self._relationship_type).available())