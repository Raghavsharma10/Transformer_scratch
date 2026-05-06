def has_child_vaults(self, vault_id):
        """Tests if a vault has any children.

        arg:    vault_id (osid.id.Id): a ``vault_id``
        return: (boolean) - ``true`` if the ``vault_id`` has children,
                ``false`` otherwise
        raise:  NotFound - ``vault_id`` is not found
        raise:  NullArgument - ``vault_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.has_child_bins
        if self._catalog_session is not None:
            return self._catalog_session.has_child_catalogs(catalog_id=vault_id)
        return self._hierarchy_session.has_children(id_=vault_id)