def add_root_bank(self, bank_id):
        """Adds a root bank.

        arg:    bank_id (osid.id.Id): the ``Id`` of a bank
        raise:  AlreadyExists - ``bank_id`` is already in hierarchy
        raise:  NotFound - ``bank_id`` not found
        raise:  NullArgument - ``bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchyDesignSession.add_root_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.add_root_catalog(catalog_id=bank_id)
        return self._hierarchy_session.add_root(id_=bank_id)