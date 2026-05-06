def get_root_vaults(self):
        """Gets the root vaults in this vault hierarchy.

        return: (osid.authorization.VaultList) - the root vaults
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method is must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_root_bins
        if self._catalog_session is not None:
            return self._catalog_session.get_root_catalogs()
        return VaultLookupSession(
            self._proxy,
            self._runtime).get_vaults_by_ids(list(self.get_root_vault_ids()))