def has_parent_banks(self, bank_id):
        """Tests if the ``Bank`` has any parents.

        arg:    bank_id (osid.id.Id): a bank ``Id``
        return: (boolean) - ``true`` if the bank has parents, ``false``
                otherwise
        raise:  NotFound - ``bank_id`` is not found
        raise:  NullArgument - ``bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.has_parent_bins
        if self._catalog_session is not None:
            return self._catalog_session.has_parent_catalogs(catalog_id=bank_id)
        return self._hierarchy_session.has_parents(id_=bank_id)