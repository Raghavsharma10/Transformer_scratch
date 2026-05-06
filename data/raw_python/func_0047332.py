def get_roots(self):
        """Gets the root nodes of this hierarchy.

        return: (osid.id.IdList) - the root nodes
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        id_list = []
        for r in self._rls.get_relationships_by_genus_type_for_source(self._phantom_root_id, self._relationship_type):
            id_list.append(r.get_destination_id())
        return IdList(id_list)