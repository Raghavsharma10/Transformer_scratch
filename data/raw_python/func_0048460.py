def get_catalog_nodes(self, catalog_id, ancestor_levels, descendant_levels, include_siblings):
        """Gets a portion of the hierarchy for the given catalog.

        arg:    catalog_id (osid.id.Id): the ``Id`` to query
        arg:    ancestor_levels (cardinal): the maximum number of
                ancestor levels to include. A value of 0 returns no
                parents in the node.
        arg:    descendant_levels (cardinal): the maximum number of
                descendant levels to include. A value of 0 returns no
                children in the node.
        arg:    include_siblings (boolean): ``true`` to include the
                siblings of the given node, ``false`` to omit the
                siblings
        return: (osid.cataloging.CatalogNode) - a catalog node
        raise:  NotFound - a ``Catalog`` identified by ``Id is`` not
                found
        raise:  NullArgument - ``catalog_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_bin_nodes
        return objects.CatalogNode(self.get_catalog_node_ids(
            catalog_id=catalog_id,
            ancestor_levels=ancestor_levels,
            descendant_levels=descendant_levels,
            include_siblings=include_siblings)._my_map, runtime=self._runtime, proxy=self._proxy)