def get_resource_ids_by_bins(self, bin_ids):
        """Gets the list of ``Resource Ids`` corresponding to a list of ``Bin`` objects.

        arg:    bin_ids (osid.id.IdList): list of bin ``Ids``
        return: (osid.id.IdList) - list of resource ``Ids``
        raise:  NullArgument - ``bin_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resource_ids_by_bins
        id_list = []
        for resource in self.get_resources_by_bins(bin_ids):
            id_list.append(resource.get_id())
        return IdList(id_list)