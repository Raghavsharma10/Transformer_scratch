def get_resources_by_bins(self, bin_ids):
        """Gets the list of ``Resources`` corresponding to a list of ``Bins``.

        arg:    bin_ids (osid.id.IdList): list of bin ``Ids``
        return: (osid.resource.ResourceList) - list of resources
        raise:  NullArgument - ``bin_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resources_by_bins
        resource_list = []
        for bin_id in bin_ids:
            resource_list += list(
                self.get_resources_by_bin(bin_id))
        return objects.ResourceList(resource_list)