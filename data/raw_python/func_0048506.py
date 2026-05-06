def get_vault(self):
        """Gets the ``Vault`` at this node.

        return: (osid.authorization.Vault) - the vault represented by
                this node
        *compliance: mandatory -- This method must be implemented.*

        """
        if self._lookup_session is None:
            mgr = get_provider_manager('AUTHORIZATION', runtime=self._runtime, proxy=self._proxy)
            self._lookup_session = mgr.get_vault_lookup_session(proxy=getattr(self, "_proxy", None))
        return self._lookup_session.get_vault(Id(self._my_map['id']))