def has_parent_bins(self, bin_id):
        """Tests if the ``Bin`` has any parents.

        arg:    bin_id (osid.id.Id): the ``Id`` of a bin
        return: (boolean) - ``true`` if the bin has parents, ``false``
                otherwise
        raise:  NotFound - ``bin_id`` is not found
        raise:  NullArgument - ``bin_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.has_parent_bins
        if self._catalog_session is not None:
            return self._catalog_session.has_parent_catalogs(catalog_id=bin_id)
        return self._hierarchy_session.has_parents(id_=bin_id)