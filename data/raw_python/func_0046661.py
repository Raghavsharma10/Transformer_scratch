def get_authorizations_ids_by_vault(self, vault_ids):
        """Gets the list of ``Authorization Ids`` corresponding to a list of ``Vault`` objects.

        arg:    vault_ids (osid.id.IdList): list of vault ``Ids``
        return: (osid.id.IdList) - list of authorization ``Ids``
        raise:  NullArgument - ``vault_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resource_ids_by_bin
        id_list = []
        for authorization in self.get_authorizations_by_vault(vault_ids):
            id_list.append(authorization.get_id())
        return IdList(id_list)