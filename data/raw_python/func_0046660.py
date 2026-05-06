def get_authorizations_by_vault(self, vault_id):
        """Gets the list of ``Authorizations`` associated with a ``Vault``.

        arg:    vault_id (osid.id.Id): ``Id`` of a ``Vault``
        return: (osid.authorization.AuthorizationList) - list of related
                authorization
        raise:  NotFound - ``vault_id`` is not found
        raise:  NullArgument - ``vault_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resources_by_bin
        mgr = self._get_provider_manager('AUTHORIZATION', local=True)
        lookup_session = mgr.get_authorization_lookup_session_for_vault(vault_ids, proxy=self._proxy)
        lookup_session.use_isolated_vault_view()
        return lookup_session.get_authorizations()