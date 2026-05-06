def get_bin_ids_by_resource(self, resource_id):
        """Gets the list of ``Bin``  ``Ids`` mapped to a ``Resource``.

        arg:    resource_id (osid.id.Id): ``Id`` of a ``Resource``
        return: (osid.id.IdList) - list of bin ``Ids``
        raise:  NotFound - ``resource_id`` is not found
        raise:  NullArgument - ``resource_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_bin_ids_by_resource
        mgr = self._get_provider_manager('RESOURCE', local=True)
        lookup_session = mgr.get_resource_lookup_session(proxy=self._proxy)
        lookup_session.use_federated_bin_view()
        resource = lookup_session.get_resource(resource_id)
        id_list = []
        for idstr in resource._my_map['assignedBinIds']:
            id_list.append(Id(idstr))
        return IdList(id_list)