def remove_root_catalog(self, catalog_id):
        """Removes a root catalog.

        arg:    catalog_id (osid.id.Id): the ``Id`` of a catalog
        raise:  NotFound - ``catalog_id`` is not a root
        raise:  NullArgument - ``catalog_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchyDesignSession.remove_root_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.remove_root_catalog(catalog_id=catalog_id)
        return self._hierarchy_session.remove_root(id_=catalog_id)