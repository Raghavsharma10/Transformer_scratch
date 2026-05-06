def remove_root(self, id_):
        """Removes a root node.

        arg:    id (osid.id.Id): the ``Id`` of the node
        raise:  NotFound - ``id`` was not found or not in hierarchy
        raise:  NullArgument - ``id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        result = self._rls.get_relationships_by_genus_type_for_peers(self._phantom_root_id, id_, self._relationship_type)
        if not bool(result.available()):
            raise errors.NotFound()
        self._ras.delete_relationship(result.get_next_relationship().get_id())
        self._adopt_orphans(id_)