def get_parent_vault_ids(self, vault_id):
        """Gets the parent ``Ids`` of the given vault.

        arg:    vault_id (osid.id.Id): a vault ``Id``
        return: (osid.id.IdList) - the parent ``Ids`` of the vault
        raise:  NotFound - ``vault_id`` is not found
        raise:  NullArgument - ``vault_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_parent_bin_ids
        if self._catalog_session is not None:
            return self._catalog_session.get_parent_catalog_ids(catalog_id=vault_id)
        return self._hierarchy_session.get_parents(id_=vault_id)