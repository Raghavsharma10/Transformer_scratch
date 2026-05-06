def get_root_objective_banks(self):
        """Gets the root objective banks in this objective bank hierarchy.

        return: (osid.learning.ObjectiveBankList) - the root objective
                banks
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method is must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_root_bins
        if self._catalog_session is not None:
            return self._catalog_session.get_root_catalogs()
        return ObjectiveBankLookupSession(
            self._proxy,
            self._runtime).get_objective_banks_by_ids(list(self.get_root_objective_bank_ids()))