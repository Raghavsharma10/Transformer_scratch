def add_root(self, id_):
        """Adds a root node.

        arg:    id (osid.id.Id): the ``Id`` of the node
        raise:  AlreadyExists - ``id`` is already in hierarchy
        raise:  NotFound - ``id`` not found
        raise:  NullArgument - ``id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        if (bool(self._rls.get_relationships_by_genus_type_for_source(id_, self._relationship_type).available()) or
                bool(self._rls.get_relationships_by_genus_type_for_destination(id_, self._relationship_type).available())):
            raise errors.AlreadyExists()
        self._assign_as_root(id_)