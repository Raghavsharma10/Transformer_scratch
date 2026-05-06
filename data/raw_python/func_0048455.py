def has_child_catalogs(self, catalog_id):
        """Tests if a catalog has any children.

        arg:    catalog_id (osid.id.Id): a ``catalog_id``
        return: (boolean) - ``true`` if the ``catalog_id`` has children,
                ``false`` otherwise
        raise:  NotFound - ``catalog_id`` is not found
        raise:  NullArgument - ``catalog_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.has_child_bins
        if self._catalog_session is not None:
            return self._catalog_session.has_child_catalogs(catalog_id=catalog_id)
        return self._hierarchy_session.has_children(id_=catalog_id)