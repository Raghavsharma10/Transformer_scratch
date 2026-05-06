def remove_root_bank(self, bank_id):
        """Removes a root bank from this hierarchy.

        arg:    bank_id (osid.id.Id): the ``Id`` of a bank
        raise:  NotFound - ``bank_id`` not a parent of ``child_id``
        raise:  NullArgument - ``bank_id`` or ``child_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchyDesignSession.remove_root_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.remove_root_catalog(catalog_id=bank_id)
        return self._hierarchy_session.remove_root(id_=bank_id)