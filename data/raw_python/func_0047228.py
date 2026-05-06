def remove_child_objective_banks(self, objective_bank_id):
        """Removes all children from an objective bank.

        arg:    objective_bank_id (osid.id.Id): the ``Id`` of an
                objective bank
        raise:  NotFound - ``objective_bank_id`` not in hierarchy
        raise:  NullArgument - ``objective_bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchyDesignSession.remove_child_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.remove_child_catalogs(catalog_id=objective_bank_id)
        return self._hierarchy_session.remove_children(id_=objective_bank_id)