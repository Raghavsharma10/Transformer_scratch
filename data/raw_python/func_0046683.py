def is_child_of_vault(self, id_, vault_id):
        """Tests if a vault is a direct child of another.

        arg:    id (osid.id.Id): an ``Id``
        arg:    vault_id (osid.id.Id): the ``Id`` of a vault
        return: (boolean) - ``true`` if the ``id`` is a child of
                ``vault_id,``  ``false`` otherwise
        raise:  NotFound - ``vault_id`` not found
        raise:  NullArgument - ``vault_id`` or ``id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*
        *implementation notes*: If ``id`` not found return ``false``.

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.is_child_of_bin
        if self._catalog_session is not None:
            return self._catalog_session.is_child_of_catalog(id_=id_, catalog_id=vault_id)
        return self._hierarchy_session.is_child(id_=vault_id, child_id=id_)