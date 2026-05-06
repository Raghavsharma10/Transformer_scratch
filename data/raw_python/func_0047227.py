def remove_child_objective_bank(self, objective_bank_id, child_id):
        """Removes a child from an objective bank.

        arg:    objective_bank_id (osid.id.Id): the ``Id`` of an
                objective bank
        arg:    child_id (osid.id.Id): the ``Id`` of the child
        raise:  NotFound - ``objective_bank_id`` not a parent of
                ``child_id``
        raise:  NullArgument - ``objective_bank_id`` or ``child_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchyDesignSession.remove_child_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.remove_child_catalog(catalog_id=objective_bank_id, child_id=child_id)
        return self._hierarchy_session.remove_child(id_=objective_bank_id, child_id=child_id)