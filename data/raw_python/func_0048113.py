def remove_child_banks(self, bank_id):
        """Removes all children from a bank.

        arg:    bank_id (osid.id.Id): the ``Id`` of a bank
        raise:  NotFound - ``bank_id`` is not in hierarchy
        raise:  NullArgument - ``bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchyDesignSession.remove_child_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.remove_child_catalogs(catalog_id=bank_id)
        return self._hierarchy_session.remove_children(id_=bank_id)