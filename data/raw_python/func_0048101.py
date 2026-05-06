def get_parent_banks(self, bank_id):
        """Gets the parents of the given bank.

        arg:    bank_id (osid.id.Id): a bank ``Id``
        return: (osid.assessment.BankList) - the parents of the bank
        raise:  NotFound - ``bank_id`` is not found
        raise:  NullArgument - ``bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_parent_bins
        if self._catalog_session is not None:
            return self._catalog_session.get_parent_catalogs(catalog_id=bank_id)
        return BankLookupSession(
            self._proxy,
            self._runtime).get_banks_by_ids(
                list(self.get_parent_bank_ids(bank_id)))