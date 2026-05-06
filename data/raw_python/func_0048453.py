def get_parent_catalogs(self, catalog_id):
        """Gets the parent catalogs of the given ``id``.

        arg:    catalog_id (osid.id.Id): the ``Id`` of the ``Catalog``
                to query
        return: (osid.cataloging.CatalogList) - the parent catalogs of
                the ``id``
        raise:  NotFound - a ``Catalog`` identified by ``Id is`` not
                found
        raise:  NullArgument - ``catalog_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_parent_bins
        if self._catalog_session is not None:
            return self._catalog_session.get_parent_catalogs(catalog_id=catalog_id)
        return CatalogLookupSession(
            self._proxy,
            self._runtime).get_catalogs_by_ids(
                list(self.get_parent_catalog_ids(catalog_id)))