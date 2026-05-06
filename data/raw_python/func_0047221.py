def get_child_objective_banks(self, objective_bank_id):
        """Gets the children of the given objective bank.

        arg:    objective_bank_id (osid.id.Id): the ``Id`` to query
        return: (osid.learning.ObjectiveBankList) - the children of the
                objective bank
        raise:  NotFound - ``objective_bank_id`` is not found
        raise:  NullArgument - ``objective_bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_child_bins
        if self._catalog_session is not None:
            return self._catalog_session.get_child_catalogs(catalog_id=objective_bank_id)
        return ObjectiveBankLookupSession(
            self._proxy,
            self._runtime).get_objective_banks_by_ids(
                list(self.get_child_objective_bank_ids(objective_bank_id)))