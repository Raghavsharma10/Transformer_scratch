def get_child_family_ids(self, family_id):
        """Gets the child ``Ids`` of the given family.

        arg:    family_id (osid.id.Id): the ``Id`` to query
        return: (osid.id.IdList) - the children of the family
        raise:  NotFound - ``family_id`` is not found
        raise:  NullArgument - ``family_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_child_bin_ids
        if self._catalog_session is not None:
            return self._catalog_session.get_child_catalog_ids(catalog_id=family_id)
        return self._hierarchy_session.get_children(id_=family_id)