def get_vaults_by_query(self, vault_query):
        """Gets a list of ``Vault`` objects matching the given search.

        arg:    vault_query (osid.authorization.VaultQuery): the vault
                query
        return: (osid.authorization.VaultList) - the returned
                ``VaultList``
        raise:  NullArgument - ``vault_query`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``vault_query`` is not of this service
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinQuerySession.get_bins_by_query_template
        if self._catalog_session is not None:
            return self._catalog_session.get_catalogs_by_query(vault_query)
        query_terms = dict(vault_query._query_terms)
        collection = JSONClientValidated('authorization',
                                         collection='Vault',
                                         runtime=self._runtime)
        result = collection.find(query_terms).sort('_id', DESCENDING)

        return objects.VaultList(result, runtime=self._runtime)