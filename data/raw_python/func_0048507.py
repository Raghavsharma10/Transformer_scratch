def get_parent_vault_nodes(self):
        """Gets the parents of this vault.

        return: (osid.authorization.VaultNodeList) - the parents of this
                vault
        *compliance: mandatory -- This method must be implemented.*

        """
        parent_vault_nodes = []
        for node in self._my_map['parentNodes']:
            parent_vault_nodes.append(VaultNode(
                node._my_map,
                runtime=self._runtime,
                proxy=self._proxy,
                lookup_session=self._lookup_session))
        return VaultNodeList(parent_vault_nodes)