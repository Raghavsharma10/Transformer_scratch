def get_parent_catalog_ids(self, catalog_id):
        """Gets the parent ``Ids`` of the given catalog.

        arg:    catalog_id (osid.id.Id): a catalog ``Id``
        return: (osid.id.IdList) - the parent ``Ids`` of the catalog
        raise:  NotFound - ``catalog_id`` is not found
        raise:  NullArgument - ``catalog_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_parent_bin_ids
        if self._catalog_session is not None:
            return self._catalog_session.get_parent_catalog_ids(catalog_id=catalog_id)
        return self._hierarchy_session.get_parents(id_=catalog_id)