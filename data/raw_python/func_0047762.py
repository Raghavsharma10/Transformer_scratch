def get_resource_ids_by_bin(self, bin_id):
        """Gets the list of ``Resource``  ``Ids`` associated with a ``Bin``.

        arg:    bin_id (osid.id.Id): ``Id`` of a ``Bin``
        return: (osid.id.IdList) - list of related resource ``Ids``
        raise:  NotFound - ``bin_id`` is not found
        raise:  NullArgument - ``bin_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resource_ids_by_bin
        id_list = []
        for resource in self.get_resources_by_bin(bin_id):
            id_list.append(resource.get_id())
        return IdList(id_list)