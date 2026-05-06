def get_bins_by_resource(self, resource_id):
        """Gets the list of ``Bin`` objects mapped to a ``Resource``.

        arg:    resource_id (osid.id.Id): ``Id`` of a ``Resource``
        return: (osid.resource.BinList) - list of bins
        raise:  NotFound - ``resource_id`` is not found
        raise:  NullArgument - ``resource_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_bins_by_resource
        mgr = self._get_provider_manager('RESOURCE', local=True)
        lookup_session = mgr.get_bin_lookup_session(proxy=self._proxy)
        return lookup_session.get_bins_by_ids(
            self.get_bin_ids_by_resource(resource_id))