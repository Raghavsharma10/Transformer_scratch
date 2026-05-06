def unassign_authorization_from_vault(self, authorization_id, vault_id):
        """Removes an ``Authorization`` from a ``Vault``.

        arg:    authorization_id (osid.id.Id): the ``Id`` of the
                ``Authorization``
        arg:    vault_id (osid.id.Id): the ``Id`` of the ``Vault``
        raise:  NotFound - ``authorization_id`` or ``vault_id`` not
                found or ``authorization_id`` not assigned to
                ``vault_id``
        raise:  NullArgument - ``authorization_id`` or ``vault_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinAssignmentSession.unassign_resource_from_bin
        mgr = self._get_provider_manager('AUTHORIZATION', local=True)
        lookup_session = mgr.get_vault_lookup_session(proxy=self._proxy)
        lookup_session.get_vault(vault_id)  # to raise NotFound
        self._unassign_object_from_catalog(authorization_id, vault_id)