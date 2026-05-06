def get_parent_catalog_nodes(self):
        """Gets the parents of this catalog.

        return: (osid.cataloging.CatalogNodeList) - the parents of the
                ``id``
        *compliance: mandatory -- This method must be implemented.*

        """
        parent_catalog_nodes = []
        for node in self._my_map['parentNodes']:
            parent_catalog_nodes.append(CatalogNode(
                node._my_map,
                runtime=self._runtime,
                proxy=self._proxy,
                lookup_session=self._lookup_session))
        return CatalogNodeList(parent_catalog_nodes)