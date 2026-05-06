def is_parent(self, id_, parent_id):
        """Tests if an ``Id`` is a direct parent of another.

        arg:    id (osid.id.Id): the ``Id`` to query
        arg:    parent_id (osid.id.Id): the ``Id`` of a parent
        return: (boolean) - ``true`` if this ``parent_id`` is a parent
                of ``id,``  ``false`` otherwise
        raise:  NotFound - ``id`` is not found
        raise:  NullArgument - ``id`` or ``parent_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*
        *implementation notes*: If ``parent_id`` not found return
        ``false``.

        """
        return bool(self._rls.get_relationships_by_genus_type_for_peers(
            parent_id,
            id_,
            self._relationship_type).available())