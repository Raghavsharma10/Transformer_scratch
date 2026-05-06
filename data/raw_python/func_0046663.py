def get_assignable_vault_ids(self, vault_id):
        """Gets a list of vault including and under the given vault node in which any authorization can be assigned.

        arg:    vault_id (osid.id.Id): the ``Id`` of the ``Vault``
        return: (osid.id.IdList) - list of assignable vault ``Ids``
        raise:  NullArgument - ``vault_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinAssignmentSession.get_assignable_bin_ids
        # This will likely be overridden by an authorization adapter
        mgr = self._get_provider_manager('AUTHORIZATION', local=True)
        lookup_session = mgr.get_vault_lookup_session(proxy=self._proxy)
        vaults = lookup_session.get_vaults()
        id_list = []
        for vault in vaults:
            id_list.append(vault.get_id())
        return IdList(id_list)