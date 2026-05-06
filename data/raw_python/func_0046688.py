def add_root_vault(self, vault_id):
        """Adds a root vault.

        arg:    vault_id (osid.id.Id): the ``Id`` of a vault
        raise:  AlreadyExists - ``vault_id`` is already in hierarchy
        raise:  NotFound - ``vault_id`` not found
        raise:  NullArgument - ``vault_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchyDesignSession.add_root_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.add_root_catalog(catalog_id=vault_id)
        return self._hierarchy_session.add_root(id_=vault_id)