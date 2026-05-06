def remove_child(self, id_, child_id):
        """Removes a childfrom an ``Id``.

        arg:    id (osid.id.Id): the ``Id`` of the node
        arg:    child_id (osid.id.Id): the ``Id`` of the child to remove
        raise:  NotFound - ``id`` or ``child_id`` was not found or
                ``child_id`` is not a child of ``id``
        raise:  NullArgument - ``id`` or ``child_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        result = self._rls.get_relationships_by_genus_type_for_peers(id_, child_id, self._relationship_type)
        if not bool(result.available()):
            raise errors.NotFound()
        self._ras.delete_relationship(result.get_next_relationship().get_id())