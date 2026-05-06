def reassign_authorization_to_vault(self, authorization_id, from_vault_id, to_vault_id):
        """Moves an ``Authorization`` from one ``Vault`` to another.

        Mappings to other ``Vaults`` are unaffected.

        arg:    authorization_id (osid.id.Id): the ``Id`` of the
                ``Authorization``
        arg:    from_vault_id (osid.id.Id): the ``Id`` of the current
                ``Vault``
        arg:    to_vault_id (osid.id.Id): the ``Id`` of the destination
                ``Vault``
        raise:  NotFound - ``authorization_id, from_vault_id,`` or
                ``to_vault_id`` not found or ``authorization_id`` not
                mapped to ``from_vault_id``
        raise:  NullArgument - ``authorization_id, from_vault_id,`` or
                ``to_vault_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinAssignmentSession.reassign_resource_to_bin
        self.assign_authorization_to_vault(authorization_id, to_vault_id)
        try:
            self.unassign_authorization_from_vault(authorization_id, from_vault_id)
        except:  # something went wrong, roll back assignment to to_vault_id
            self.unassign_authorization_from_vault(authorization_id, to_vault_id)
            raise