def get_child_bins(self, bin_id):
        """Gets the children of the given bin.

        arg:    bin_id (osid.id.Id): the ``Id`` to query
        return: (osid.resource.BinList) - the children of the bin
        raise:  NotFound - ``bin_id`` not found
        raise:  NullArgument - ``bin_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_child_bins
        if self._catalog_session is not None:
            return self._catalog_session.get_child_catalogs(catalog_id=bin_id)
        return BinLookupSession(
            self._proxy,
            self._runtime).get_bins_by_ids(
                list(self.get_child_bin_ids(bin_id)))