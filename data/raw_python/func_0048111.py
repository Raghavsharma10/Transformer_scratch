def add_child_bank(self, bank_id, child_id):
        """Adds a child to a bank.

        arg:    bank_id (osid.id.Id): the ``Id`` of a bank
        arg:    child_id (osid.id.Id): the ``Id`` of the new child
        raise:  AlreadyExists - ``bank_id`` is already a parent of
                ``child_id``
        raise:  NotFound - ``bank_id`` or ``child_id`` not found
        raise:  NullArgument - ``bank_id`` or ``child_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchyDesignSession.add_child_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.add_child_catalog(catalog_id=bank_id, child_id=child_id)
        return self._hierarchy_session.add_child(id_=bank_id, child_id=child_id)