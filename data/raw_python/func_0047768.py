def get_assignable_bin_ids(self, bin_id):
        """Gets a list of bins including and under the given bin node in which any resource can be assigned.

        arg:    bin_id (osid.id.Id): the ``Id`` of the ``Bin``
        return: (osid.id.IdList) - list of assignable bin ``Ids``
        raise:  NullArgument - ``bin_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinAssignmentSession.get_assignable_bin_ids
        # This will likely be overridden by an authorization adapter
        mgr = self._get_provider_manager('RESOURCE', local=True)
        lookup_session = mgr.get_bin_lookup_session(proxy=self._proxy)
        bins = lookup_session.get_bins()
        id_list = []
        for bin in bins:
            id_list.append(bin.get_id())
        return IdList(id_list)