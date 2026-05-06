def add_root_catalog(self, catalog_id):
        """Adds a root catalog.

        arg:    catalog_id (osid.id.Id): the ``Id`` of a catalog
        raise:  AlreadyExists - ``catalog_id`` is already in hierarchy
        raise:  NotFound - ``catalog_id`` not found
        raise:  NullArgument - ``catalog_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchyDesignSession.add_root_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.add_root_catalog(catalog_id=catalog_id)
        return self._hierarchy_session.add_root(id_=catalog_id)