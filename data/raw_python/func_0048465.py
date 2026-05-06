def remove_child_catalogs(self, catalog_id):
        """Removes all children from a catalog.

        arg:    catalog_id (osid.id.Id): the ``Id`` of a catalog
        raise:  NotFound - ``catalog_id`` is not in hierarchy
        raise:  NullArgument - ``catalog_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchyDesignSession.remove_child_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.remove_child_catalogs(catalog_id=catalog_id)
        return self._hierarchy_session.remove_children(id_=catalog_id)