def get_vault_ids_by_authorization(self, authorization_id):
        """Gets the list of ``Vault``  ``Ids`` mapped to an ``Authorization``.

        arg:    authorization_id (osid.id.Id): ``Id`` of an
                ``Authorization``
        return: (osid.id.IdList) - list of vault ``Ids``
        raise:  NotFound - ``authorization_id`` is not found
        raise:  NullArgument - ``authorization_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_bin_ids_by_resource
        mgr = self._get_provider_manager('AUTHORIZATION', local=True)
        lookup_session = mgr.get_authorization_lookup_session(proxy=self._proxy)
        lookup_session.use_federated_vault_view()
        authorization = lookup_session.get_authorization(authorization_id)
        id_list = []
        for idstr in authorization._my_map['assignedVaultIds']:
            id_list.append(Id(idstr))
        return IdList(id_list)