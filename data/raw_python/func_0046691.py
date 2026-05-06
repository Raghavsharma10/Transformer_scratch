def remove_child_vault(self, vault_id, child_id):
        """Removes a child from a vault.

        arg:    vault_id (osid.id.Id): the ``Id`` of a vault
        arg:    child_id (osid.id.Id): the ``Id`` of the child
        raise:  NotFound - ``vault_id`` not parent of ``child_id``
        raise:  NullArgument - ``vault_id`` or ``child_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchyDesignSession.remove_child_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.remove_child_catalog(catalog_id=vault_id, child_id=child_id)
        return self._hierarchy_session.remove_child(id_=vault_id, child_id=child_id)