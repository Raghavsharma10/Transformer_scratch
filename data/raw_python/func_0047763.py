def get_resources_by_bin(self, bin_id):
        """Gets the list of ``Resources`` associated with a ``Bin``.

        arg:    bin_id (osid.id.Id): ``Id`` of a ``Bin``
        return: (osid.resource.ResourceList) - list of related resources
        raise:  NotFound - ``bin_id`` is not found
        raise:  NullArgument - ``bin_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resources_by_bin
        mgr = self._get_provider_manager('RESOURCE', local=True)
        lookup_session = mgr.get_resource_lookup_session_for_bin(bin_id, proxy=self._proxy)
        lookup_session.use_isolated_bin_view()
        return lookup_session.get_resources()