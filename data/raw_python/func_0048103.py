def has_child_banks(self, bank_id):
        """Tests if a bank has any children.

        arg:    bank_id (osid.id.Id): a ``bank_id``
        return: (boolean) - ``true`` if the ``bank_id`` has children,
                ``false`` otherwise
        raise:  NotFound - ``bank_id`` is not found
        raise:  NullArgument - ``bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.has_child_bins
        if self._catalog_session is not None:
            return self._catalog_session.has_child_catalogs(catalog_id=bank_id)
        return self._hierarchy_session.has_children(id_=bank_id)