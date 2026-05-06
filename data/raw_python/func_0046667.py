def get_vaults_by_genus_type(self, vault_genus_type):
        """Gets a ``VaultList`` corresponding to the given vault genus ``Type`` which does not include vaults of types derived from the specified ``Type``.

        In plenary mode, the returned list contains all known vaults or
        an error results. Otherwise, the returned list may contain only
        those vaults that are accessible through this session.

        arg:    vault_genus_type (osid.type.Type): a vault genus type
        return: (osid.authorization.VaultList) - the returned ``Vault``
                list
        raise:  NullArgument - ``vault_genus_type`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        collection = JSONClientValidated('authorization',
                                         collection='Vault',
                                         runtime=self._runtime)
        result = collection.find({'genusTypeId': {'$in': [str(vault_genus_type)]}}).sort('_id', DESCENDING)

        return objects.VaultList(result, runtime=self._runtime)