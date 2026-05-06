def get_banks_by_query(self, bank_query):
        """Gets a list of ``Bank`` objects matching the given bank query.

        arg:    bank_query (osid.assessment.BankQuery): the bank query
        return: (osid.assessment.BankList) - the returned ``BankList``
        raise:  NullArgument - ``bank_query`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        raise:  Unsupported - ``bank_query`` is not of this service
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinQuerySession.get_bins_by_query_template
        if self._catalog_session is not None:
            return self._catalog_session.get_catalogs_by_query(bank_query)
        query_terms = dict(bank_query._query_terms)
        collection = JSONClientValidated('assessment',
                                         collection='Bank',
                                         runtime=self._runtime)
        result = collection.find(query_terms).sort('_id', DESCENDING)

        return objects.BankList(result, runtime=self._runtime)