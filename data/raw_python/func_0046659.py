def get_authorization_ids_by_vault(self, vault_id):
        """Gets the list of ``Authorization``  ``Ids`` associated with a ``Vault``.

        arg:    vault_id (osid.id.Id): ``Id`` of a ``Vault``
        return: (osid.id.IdList) - list of related authorization ``Ids``
        raise:  NotFound - ``vault_id`` is not found
        raise:  NullArgument - ``vault_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resource_ids_by_bin
        id_list = []
        for authorization in self.get_authorizations_by_vault(vault_id):
            id_list.append(authorization.get_id())
        return IdList(id_list)