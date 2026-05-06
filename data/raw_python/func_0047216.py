def get_parent_objective_banks(self, objective_bank_id):
        """Gets the parents of the given objective bank.

        arg:    objective_bank_id (osid.id.Id): the ``Id`` of an
                objective bank
        return: (osid.learning.ObjectiveBankList) - the parents of the
                objective bank
        raise:  NotFound - ``objective_bank_id`` is not found
        raise:  NullArgument - ``objective_bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_parent_bins
        if self._catalog_session is not None:
            return self._catalog_session.get_parent_catalogs(catalog_id=objective_bank_id)
        return ObjectiveBankLookupSession(
            self._proxy,
            self._runtime).get_objective_banks_by_ids(
                list(self.get_parent_objective_bank_ids(objective_bank_id)))