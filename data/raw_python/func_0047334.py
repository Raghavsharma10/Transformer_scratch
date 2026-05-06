def get_parents(self, id_):
        """Gets the parents of the given ``id``.

        arg:    id (osid.id.Id): the ``Id`` to query
        return: (osid.id.IdList) - the parents of the ``id``
        raise:  NotFound - ``id`` is not found
        raise:  NullArgument - ``id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        id_list = []
        for r in self._rls.get_relationships_by_genus_type_for_destination(id_, self._relationship_type):
            ident = r.get_source_id()
            if ident != self._phantom_root_id:
                id_list.append(ident)
        return IdList(id_list)