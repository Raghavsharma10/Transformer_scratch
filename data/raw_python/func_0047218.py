def has_child_objective_banks(self, objective_bank_id):
        """Tests if an objective bank has any children.

        arg:    objective_bank_id (osid.id.Id): the ``Id`` of an
                objective bank
        return: (boolean) - ``true`` if the ``objective_bank_id`` has
                children, ``false`` otherwise
        raise:  NotFound - ``objective_bank_id`` is not found
        raise:  NullArgument - ``objective_bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.has_child_bins
        if self._catalog_session is not None:
            return self._catalog_session.has_child_catalogs(catalog_id=objective_bank_id)
        return self._hierarchy_session.has_children(id_=objective_bank_id)