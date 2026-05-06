def get_parent_vaults(self, vault_id):
        """Gets the parents of the given vault.

        arg:    vault_id (osid.id.Id): a vault ``Id``
        return: (osid.authorization.VaultList) - the parents of the
                vault
        raise:  NotFound - ``vault_id`` is not found
        raise:  NullArgument - ``vault_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_parent_bins
        if self._catalog_session is not None:
            return self._catalog_session.get_parent_catalogs(catalog_id=vault_id)
        return VaultLookupSession(
            self._proxy,
            self._runtime).get_vaults_by_ids(
                list(self.get_parent_vault_ids(vault_id)))