def has_parent_catalogs(self, catalog_id):
        """Tests if the ``Catalog`` has any parents.

        arg:    catalog_id (osid.id.Id): a catalog ``Id``
        return: (boolean) - ``true`` if the catalog has parents,
                ``false`` otherwise
        raise:  NotFound - ``catalog_id`` is not found
        raise:  NullArgument - ``catalog_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.has_parent_bins
        if self._catalog_session is not None:
            return self._catalog_session.has_parent_catalogs(catalog_id=catalog_id)
        return self._hierarchy_session.has_parents(id_=catalog_id)