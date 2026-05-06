def has_parent_vaults(self, vault_id):
        """Tests if the ``Vault`` has any parents.

        arg:    vault_id (osid.id.Id): a vault ``Id``
        return: (boolean) - ``true`` if the vault has parents, ``false``
                otherwise
        raise:  NotFound - ``vault_id`` is not found
        raise:  NullArgument - ``vault_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.has_parent_bins
        if self._catalog_session is not None:
            return self._catalog_session.has_parent_catalogs(catalog_id=vault_id)
        return self._hierarchy_session.has_parents(id_=vault_id)