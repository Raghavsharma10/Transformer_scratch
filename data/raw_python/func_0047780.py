def get_root_bins(self):
        """Gets the root bins in the bin hierarchy.

        A node with no parents is an orphan. While all bin ``Ids`` are
        known to the hierarchy, an orphan does not appear in the
        hierarchy unless explicitly added as a root node or child of
        another node.

        return: (osid.resource.BinList) - the root bins
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method is must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_root_bins
        if self._catalog_session is not None:
            return self._catalog_session.get_root_catalogs()
        return BinLookupSession(
            self._proxy,
            self._runtime).get_bins_by_ids(list(self.get_root_bin_ids()))