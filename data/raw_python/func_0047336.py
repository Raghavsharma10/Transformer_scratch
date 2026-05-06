def get_children(self, id_):
        """Gets the children of the given ``Id``.

        arg:    id (osid.id.Id): the ``Id`` to query
        return: (osid.id.IdList) - the children of the ``id``
        raise:  NotFound - ``id`` is not found
        raise:  NullArgument - ``id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        id_list = []
        for r in self._rls.get_relationships_by_genus_type_for_source(id_, self._relationship_type):
            id_list.append(r.get_destination_id())
        return IdList(id_list)