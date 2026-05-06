def get_child_catalog_ids(self, catalog_id):
        """Gets the child ``Ids`` of the given catalog.

        arg:    catalog_id (osid.id.Id): the ``Id`` to query
        return: (osid.id.IdList) - the children of the catalog
        raise:  NotFound - ``catalog_id`` is not found
        raise:  NullArgument - ``catalog_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_child_bin_ids
        if self._catalog_session is not None:
            return self._catalog_session.get_child_catalog_ids(catalog_id=catalog_id)
        return self._hierarchy_session.get_children(id_=catalog_id)