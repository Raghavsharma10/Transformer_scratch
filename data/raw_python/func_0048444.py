def get_catalogs_by_query(self, catalog_query):
        """Gets a list of ``Catalogs`` matching the given catalog query.

        arg:    catalog_query (osid.cataloging.CatalogQuery): the
                catalog query
        return: (osid.cataloging.CatalogList) - the returned
                ``CatalogList``
        raise:  NullArgument - ``catalog_query`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``catalog_query`` is not of this service
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinQuerySession.get_bins_by_query_template
        if self._catalog_session is not None:
            return self._catalog_session.get_catalogs_by_query(catalog_query)
        query_terms = dict(catalog_query._query_terms)
        collection = JSONClientValidated('cataloging',
                                         collection='Catalog',
                                         runtime=self._runtime)
        result = collection.find(query_terms).sort('_id', DESCENDING)

        return objects.CatalogList(result, runtime=self._runtime)