def get_parent_family_ids(self, family_id):
        """Gets the parent ``Ids`` of the given family.

        arg:    family_id (osid.id.Id): the ``Id`` of a family
        return: (osid.id.IdList) - the parent ``Ids`` of the family
        raise:  NotFound - ``family_id`` is not found
        raise:  NullArgument - ``family_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_parent_bin_ids
        if self._catalog_session is not None:
            return self._catalog_session.get_parent_catalog_ids(catalog_id=family_id)
        return self._hierarchy_session.get_parents(id_=family_id)