def get_child_vaults(self, vault_id):
        """Gets the children of the given vault.

        arg:    vault_id (osid.id.Id): the ``Id`` to query
        return: (osid.authorization.VaultList) - the children of the
                vault
        raise:  NotFound - ``vault_id`` is not found
        raise:  NullArgument - ``vault_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_child_bins
        if self._catalog_session is not None:
            return self._catalog_session.get_child_catalogs(catalog_id=vault_id)
        return VaultLookupSession(
            self._proxy,
            self._runtime).get_vaults_by_ids(
                list(self.get_child_vault_ids(vault_id)))