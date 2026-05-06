def add_root_family(self, family_id):
        """Adds a root family.

        arg:    family_id (osid.id.Id): the ``Id`` of a family
        raise:  AlreadyExists - ``family_id`` is already in hierarchy
        raise:  NotFound - ``family_id`` not found
        raise:  NullArgument - ``family_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchyDesignSession.add_root_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.add_root_catalog(catalog_id=family_id)
        return self._hierarchy_session.add_root(id_=family_id)