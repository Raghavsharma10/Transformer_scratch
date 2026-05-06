def get_bins_by_query(self, bin_query):
        """Gets a list of ``Bins`` matching the given bin query.

        arg:    bin_query (osid.resource.BinQuery): the bin query
        return: (osid.resource.BinList) - the returned ``BinList``
        raise:  NullArgument - ``bin_query`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - a ``bin_query`` is not of this service
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinQuerySession.get_bins_by_query_template
        if self._catalog_session is not None:
            return self._catalog_session.get_catalogs_by_query(bin_query)
        query_terms = dict(bin_query._query_terms)
        collection = JSONClientValidated('resource',
                                         collection='Bin',
                                         runtime=self._runtime)
        result = collection.find(query_terms).sort('_id', DESCENDING)

        return objects.BinList(result, runtime=self._runtime)