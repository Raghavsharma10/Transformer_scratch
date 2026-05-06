def get_objective_banks(self):
        """Gets all ``ObjectiveBanks``.

        In plenary mode, the returned list contains all known objective
        banks or an error results. Otherwise, the returned list may
        contain only those objective banks that are accessible through
        this session.

        return: (osid.learning.ObjectiveBankList) - a
                ``ObjectiveBankList``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinLookupSession.get_bins_template
        # NOTE: This implementation currently ignores plenary view
        if self._catalog_session is not None:
            return self._catalog_session.get_catalogs()
        collection = JSONClientValidated('learning',
                                         collection='ObjectiveBank',
                                         runtime=self._runtime)
        result = collection.find().sort('_id', DESCENDING)

        return objects.ObjectiveBankList(result, runtime=self._runtime, proxy=self._proxy)