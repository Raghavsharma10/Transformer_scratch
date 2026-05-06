def get_root_banks(self):
        """Gets the root banks in this bank hierarchy.

        return: (osid.assessment.BankList) - the root banks
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method is must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_root_bins
        if self._catalog_session is not None:
            return self._catalog_session.get_root_catalogs()
        return BankLookupSession(
            self._proxy,
            self._runtime).get_banks_by_ids(list(self.get_root_bank_ids()))