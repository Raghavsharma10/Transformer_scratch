def get_catalog(self):
        """Gets the ``Catalog`` at this node.

        return: (osid.cataloging.Catalog) - the catalog represented by
                this node
        *compliance: mandatory -- This method must be implemented.*

        """
        if self._lookup_session is None:
            mgr = get_provider_manager('CATALOGING', runtime=self._runtime, proxy=self._proxy)
            self._lookup_session = mgr.get_catalog_lookup_session(proxy=getattr(self, "_proxy", None))
        return self._lookup_session.get_catalog(Id(self._my_map['id']))