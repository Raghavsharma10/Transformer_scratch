def get_parent_bank_ids(self, bank_id):
        """Gets the parent ``Ids`` of the given bank.

        arg:    bank_id (osid.id.Id): a bank ``Id``
        return: (osid.id.IdList) - the parent ``Ids`` of the bank
        raise:  NotFound - ``bank_id`` is not found
        raise:  NullArgument - ``bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_parent_bin_ids
        if self._catalog_session is not None:
            return self._catalog_session.get_parent_catalog_ids(catalog_id=bank_id)
        return self._hierarchy_session.get_parents(id_=bank_id)